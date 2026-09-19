"""
02_compute_sue.py
=================
Module 2 of the PEAD Alpha Pipeline.

Purpose:
    Compute Standardized Unexpected Earnings (SUE) for each earnings event
    using a time-series expectation model (no analyst consensus data needed).

SUE definition:
    SUE_t = (EPS_t - E[EPS_t]) / StdDev(forecast_errors)

Expectation model — Seasonal Random Walk (primary):
    E[EPS_t] = EPS_{t-4}   (same quarter one year ago)
    Error    = EPS_t - EPS_{t-4}
    StdDev   = rolling std of last 8 quarterly errors

    This follows Bernard & Thomas (1990) — simple, robust, avoids look-ahead.

Fallback — AR(1) on seasonally-differenced series:
    Used when >= 12 quarters of history are available.
    E[dEPS_t] = alpha + beta * dEPS_{t-1}
    where dEPS_t = EPS_t - EPS_{t-4}

PIT discipline:
    - EPS history is trimmed to only quarters announced BEFORE the current
      announcement date (using announce_date from Module 1, not fiscal end date).
    - This prevents any forward-looking bias.

Data source:
    yfinance quarterly financials (EPS = net income / diluted shares)
    yfinance earningsHistory (reported EPS per quarter, cleaner)

Output:
    data/signals/sue_panel.csv
    Columns: ticker, announce_date, fiscal_period_end,
             eps_actual, eps_expected, sue, sue_model, n_quarters_used

Author: Tingxuan Wu
"""

import warnings
import requests
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).parent.parent
DATA_DIR    = _ROOT / "data"
SIGNALS_DIR = DATA_DIR / "signals"
EDGAR_CSV   = DATA_DIR / "edgar" / "earnings_dates.csv"
OUTPUT_CSV  = SIGNALS_DIR / "sue_panel.csv"

SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_QUARTERS_SRW = 5   # Seasonal Random Walk: need at least 5 quarters (4 for lag + 1 for std)
MIN_QUARTERS_AR1 = 12  # AR(1): need at least 12 quarters for reliable fit
STD_FLOOR        = 0.01  # Floor for std dev to avoid division by near-zero



# ── CIK cache (loaded once per session) ──────────────────────────────────────
_CIK_CACHE = {}

def _get_cik(ticker: str) -> str:
    if not _CIK_CACHE:
        r = requests.get("https://www.sec.gov/files/company_tickers.json",
                         headers={"User-Agent": "PEAD Research tw3196@nyu.edu"}, timeout=15)
        for entry in r.json().values():
            _CIK_CACHE[entry["ticker"].upper()] = str(entry["cik_str"]).zfill(10)
    return _CIK_CACHE.get(ticker.upper(), "")


def fetch_eps_history(ticker: str):
    """
    Fetch quarterly EPS from SEC EDGAR XBRL companyfacts.
    Returns DataFrame: fiscal_period_end, eps, filed_date
    filed_date = date this EPS value first became public (PIT anchor).
    """
    import requests as _req
    cik = _get_cik(ticker)
    if not cik:
        log.warning(f"{ticker}: CIK not found")
        return None
    try:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        r = _req.get(url, headers={"User-Agent": "PEAD Research tw3196@nyu.edu"}, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        rows = (data.get("facts", {})
                    .get("us-gaap", {})
                    .get("EarningsPerShareDiluted", {})
                    .get("units", {})
                    .get("USD/shares", []))
        if not rows:
            return None

        records = []
        for row in rows:
            fp   = row.get("fp", "")
            form = row.get("form", "")
            # Keep only true quarterly disclosures (Q1/Q2/Q3) and 10-K annual
            if form not in ("10-Q", "10-K") or not fp:
                continue
            # Exclude cumulative YTD rows: for 10-Q only keep Q1/Q2/Q3
            if form == "10-Q" and fp not in ("Q1", "Q2", "Q3"):
                continue
            # 10-K -> Q4 (annual = Q4 standalone)
            records.append({
                "fiscal_period_end": pd.Timestamp(row["end"]),
                "filed_date":        pd.Timestamp(row["filed"]),
                "eps":               float(row["val"]),
                "fp":                fp,
                "form":              form,
            })

        if not records:
            return None

        df = pd.DataFrame(records)
        # Deduplicate: same period may appear in multiple filings (amendments).
        # Keep EARLIEST filed_date (first public disclosure = best PIT).
        df = (df.sort_values("filed_date")
                .drop_duplicates(subset=["fiscal_period_end"], keep="first")
                .sort_values("fiscal_period_end")
                .reset_index(drop=True))

        log.debug(f"{ticker}: XBRL → {len(df)} quarterly EPS rows, "
                  f"{df['fiscal_period_end'].min().year}–{df['fiscal_period_end'].max().year}")
        return df

    except Exception as e:
        log.warning(f"{ticker}: XBRL fetch error — {e}")
        return None


# ── Step 2: Seasonal Random Walk expectation ──────────────────────────────────

def sue_seasonal_rw(eps_history: pd.DataFrame, as_of_date: pd.Timestamp) -> dict:
    """
    Compute SUE using Seasonal Random Walk (Bernard & Thomas 1990).

    E[EPS_t] = EPS_{t-4}  (same quarter last year)
    Error_t  = EPS_t - EPS_{t-4}
    SUE      = Error_t / rolling_std(last 8 errors)

    PIT: only use quarters with fiscal_period_end strictly before as_of_date.
    Returns dict: eps_expected, sue, std_error, n_quarters, model
    """
    # PIT cut: only history known before this announcement
    col = "filed_date" if "filed_date" in eps_history.columns else "fiscal_period_end"
    hist = eps_history[eps_history[col] < as_of_date].copy()
    hist = hist.sort_values("fiscal_period_end").reset_index(drop=True)

    if len(hist) < MIN_QUARTERS_SRW:
        return {"eps_expected": np.nan, "sue": np.nan,
                "std_error": np.nan, "n_quarters": len(hist), "model": "insufficient_data"}

    # Most recent quarter in history = t-1 announcement (current is t)
    # For seasonal RW: expected = EPS 4 quarters ago
    eps_series = hist["eps"].values

    if len(eps_series) < 4:
        return {"eps_expected": np.nan, "sue": np.nan,
                "std_error": np.nan, "n_quarters": len(hist), "model": "insufficient_data"}

    eps_expected = eps_series[-4]   # same quarter one year ago

    # Rolling errors: eps_t - eps_{t-4} for last 8 available quarters
    errors = []
    for i in range(4, len(eps_series)):
        errors.append(eps_series[i] - eps_series[i - 4])

    if len(errors) == 0:
        return {"eps_expected": eps_expected, "sue": np.nan,
                "std_error": np.nan, "n_quarters": len(hist), "model": "insufficient_data"}

    std_err = max(np.std(errors[-8:], ddof=1) if len(errors) >= 2 else abs(errors[-1]),
                  STD_FLOOR)

    return {
        "eps_expected": eps_expected,
        "std_error":    std_err,
        "n_quarters":   len(hist),
        "model":        "seasonal_rw",
    }


# ── Step 3: AR(1) expectation (when sufficient history) ──────────────────────

def sue_ar1(eps_history: pd.DataFrame, as_of_date: pd.Timestamp) -> dict:
    """
    Compute expected EPS using AR(1) on seasonally-differenced series.

    dEPS_t = EPS_t - EPS_{t-4}
    Model:  dEPS_t = alpha + beta * dEPS_{t-1}
    Fitted on all available history; E[dEPS_t] = alpha + beta * dEPS_{t-1}
    E[EPS_t] = EPS_{t-4} + E[dEPS_t]

    Returns dict same structure as sue_seasonal_rw.
    """
    col = "filed_date" if "filed_date" in eps_history.columns else "fiscal_period_end"
    hist = eps_history[eps_history[col] < as_of_date].copy()
    hist = hist.sort_values("fiscal_period_end").reset_index(drop=True)

    if len(hist) < MIN_QUARTERS_AR1:
        return sue_seasonal_rw(eps_history, as_of_date)  # fall back

    eps = hist["eps"].values
    # Seasonal differences
    deps = eps[4:] - eps[:-4]

    if len(deps) < 8:
        return sue_seasonal_rw(eps_history, as_of_date)

    # OLS: deps[t] = alpha + beta * deps[t-1]
    y = deps[1:]
    x = deps[:-1]
    # Closed-form OLS
    x_dm = x - x.mean()
    if x_dm @ x_dm < 1e-10:
        return sue_seasonal_rw(eps_history, as_of_date)

    beta  = (x_dm @ y) / (x_dm @ x_dm)
    alpha = y.mean() - beta * x.mean()
    beta  = np.clip(beta, -0.99, 0.99)  # stationarity guard

    # Forecast: E[dEPS_t] = alpha + beta * dEPS_{t-1}
    last_deps      = deps[-1]
    exp_deps       = alpha + beta * last_deps
    eps_expected   = eps[-4] + exp_deps

    # Residuals for std
    resid  = y - (alpha + beta * x)
    std_err = max(np.std(resid, ddof=2) if len(resid) >= 3 else STD_FLOOR, STD_FLOOR)

    return {
        "eps_expected": eps_expected,
        "std_error":    std_err,
        "n_quarters":   len(hist),
        "model":        "ar1",
    }


# ── Step 4: Compute SUE for one ticker ───────────────────────────────────────

def compute_sue_for_ticker(
    ticker: str,
    announce_df: pd.DataFrame,
    use_ar1: bool = True,
) -> pd.DataFrame:
    """
    Compute SUE for all earnings events of one ticker.

    announce_df: subset of earnings_dates.csv for this ticker.
                 Must have: announce_date, fiscal_period_end, source columns.

    Returns DataFrame with SUE for each announcement date.
    """
    eps_hist = fetch_eps_history(ticker)
    if eps_hist is None or len(eps_hist) < MIN_QUARTERS_SRW:
        log.warning(f"{ticker}: insufficient EPS history")
        return pd.DataFrame()

    # Keep only Item 2.02 and 8-K events (drop 10-Q fallback for IC analysis)
    # But compute SUE for all — downstream can filter by source quality
    events = announce_df.copy()
    events["announce_date"]     = pd.to_datetime(events["announce_date"])
    events["fiscal_period_end"] = pd.to_datetime(events["fiscal_period_end"])
    events = events.sort_values("announce_date").reset_index(drop=True)

    rows = []
    for _, ev in events.iterrows():
        ann_date    = ev["announce_date"]
        period_end  = ev["fiscal_period_end"]

        # Find actual EPS: the eps_history row whose fiscal_period_end is
        # closest to this event's fiscal_period_end (within 45 days)
        diff = (eps_hist["fiscal_period_end"] - period_end).abs()
        closest_idx = diff.idxmin()
        if diff[closest_idx] > pd.Timedelta(days=45):
            log.debug(f"{ticker} {ann_date.date()}: no matching EPS (closest {diff[closest_idx].days}d)")
            continue

        eps_actual = eps_hist.loc[closest_idx, "eps"]
        if pd.isna(eps_actual):
            continue

        # Expectation model (PIT: use only history before this announcement)
        if use_ar1:
            model_out = sue_ar1(eps_hist, ann_date)
        else:
            model_out = sue_seasonal_rw(eps_hist, ann_date)

        eps_expected = model_out["eps_expected"]
        std_err      = model_out.get("std_error", np.nan)

        if pd.isna(eps_expected) or pd.isna(std_err):
            sue = np.nan
        else:
            sue = (eps_actual - eps_expected) / std_err

        rows.append({
            "ticker":            ticker,
            "announce_date":     ann_date,
            "fiscal_period_end": period_end,
            "eps_actual":        eps_actual,
            "eps_expected":      eps_expected,
            "eps_surprise":      eps_actual - eps_expected,
            "std_error":         std_err,
            "sue":               sue,
            "sue_model":         model_out.get("model", "unknown"),
            "n_quarters_used":   model_out.get("n_quarters", 0),
            "source":            ev.get("source", ""),
        })

    return pd.DataFrame(rows)


# ── Step 5: Main pipeline ─────────────────────────────────────────────────────

def build_sue_panel(
    earnings_dates_path: Path = EDGAR_CSV,
    output_path: Path         = OUTPUT_CSV,
    use_ar1: bool             = True,
    min_sue_obs: int          = 3,
) -> pd.DataFrame:
    """
    Main entry point. Reads earnings_dates.csv, computes SUE for all tickers.

    Args:
        earnings_dates_path: output of Module 1
        output_path:         where to save sue_panel.csv
        use_ar1:             use AR(1) when >= 12 quarters available, else SRW
        min_sue_obs:         minimum SUE observations to include a ticker

    Returns DataFrame with SUE panel.
    """
    if not earnings_dates_path.exists():
        raise FileNotFoundError(
            f"earnings_dates.csv not found at {earnings_dates_path}. "
            "Run 01_fetch_earnings_dates.py first."
        )

    log.info(f"Loading earnings dates from {earnings_dates_path}")
    ann_df = pd.read_csv(earnings_dates_path, parse_dates=["announce_date", "fiscal_period_end"])
    tickers = ann_df["ticker"].unique().tolist()
    log.info(f"Computing SUE for {len(tickers)} tickers, {len(ann_df)} events")
    log.info(f"Expectation model: {'AR(1) + Seasonal RW fallback' if use_ar1 else 'Seasonal RW only'}")

    all_rows = []
    for i, ticker in enumerate(tickers):
        log.info(f"[{i+1}/{len(tickers)}] {ticker}")
        sub = ann_df[ann_df["ticker"] == ticker].copy()
        result = compute_sue_for_ticker(ticker, sub, use_ar1=use_ar1)
        if not result.empty and len(result) >= min_sue_obs:
            all_rows.append(result)

    if not all_rows:
        log.error("No SUE computed — check EPS data availability")
        return pd.DataFrame()

    panel = pd.concat(all_rows, ignore_index=True)
    panel = panel.sort_values(["ticker", "announce_date"]).reset_index(drop=True)

    # ── Quality report ────────────────────────────────────────────────────────
    n       = len(panel)
    n_sue   = panel["sue"].notna().sum()
    n_ar1   = (panel["sue_model"] == "ar1").sum()
    n_srw   = (panel["sue_model"] == "seasonal_rw").sum()
    n_8k    = panel["source"].str.startswith("8K").sum()

    log.info(f"\n{'='*55}")
    log.info(f"SUE panel: {n} events, {panel['ticker'].nunique()} tickers")
    log.info(f"  SUE computed:      {n_sue}/{n} ({n_sue/max(n,1):.1%})")
    log.info(f"  AR(1) model:       {n_ar1} ({n_ar1/max(n,1):.1%})")
    log.info(f"  Seasonal RW:       {n_srw} ({n_srw/max(n,1):.1%})")
    log.info(f"  8-K sourced:       {n_8k} ({n_8k/max(n,1):.1%})")
    log.info(f"  SUE mean:          {panel['sue'].mean():.3f}")
    log.info(f"  SUE std:           {panel['sue'].std():.3f}")
    log.info(f"  SUE p5/p95:        {panel['sue'].quantile(0.05):.2f} / {panel['sue'].quantile(0.95):.2f}")

    # Winsorize SUE at 1st/99th percentile (standard practice)
    p01 = panel["sue"].quantile(0.01)
    p99 = panel["sue"].quantile(0.99)
    panel["sue_winsor"] = panel["sue"].clip(lower=p01, upper=p99)
    log.info(f"  SUE winsorized at [{p01:.2f}, {p99:.2f}]")
    log.info(f"{'='*55}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    log.info(f"Saved to {output_path}")

    return panel


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute time-series SUE from earnings dates")
    parser.add_argument("--earnings-csv", default=str(EDGAR_CSV),
                        help="Path to earnings_dates.csv from Module 1")
    parser.add_argument("--output",       default=str(OUTPUT_CSV))
    parser.add_argument("--no-ar1",       action="store_true",
                        help="Use Seasonal RW only (skip AR(1))")
    args = parser.parse_args()

    panel = build_sue_panel(
        earnings_dates_path = Path(args.earnings_csv),
        output_path         = Path(args.output),
        use_ar1             = not args.no_ar1,
    )

    if not panel.empty:
        print("\nSample SUE output:")
        cols = ["ticker", "announce_date", "fiscal_period_end",
                "eps_actual", "eps_expected", "sue", "sue_model"]
        print(panel[cols].head(20).to_string(index=False))
