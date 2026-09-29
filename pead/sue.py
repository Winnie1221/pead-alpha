"""
Step 2: standardized unexpected earnings (SUE) from SEC XBRL EPS.

SUE_t = (EPS_t − E[EPS_t]) / σ_t

Expectation models (no analyst consensus needed):
  * Seasonal random walk (Bernard & Thomas 1990): E[EPS_t] = EPS_{t−4},
    σ_t = std of the last 8 seasonal differences.
  * AR(1) on seasonal differences (Foster 1977 style), used when enough
    history exists: ΔEPS_t = α + β·ΔEPS_{t−1},  E[EPS_t] = EPS_{t−4} + E[ΔEPS_t],
    σ_t = residual std. Falls back to the seasonal random walk otherwise.

Quarterly EPS construction from XBRL companyfacts:
  * Only 3-month facts (70–110 day duration) are quarterly values. 10-Qs also
    report 6- and 9-month year-to-date EPS under the same fiscal-period tag;
    those are excluded by the duration filter.
  * 10-Ks report annual EPS only, so Q4 = FY − (Q1 + Q2 + Q3). This is the
    standard approximation (exact only if the share count is constant).
  * For restated values the earliest filing is kept (first public value).

Point-in-time rules:
  * The history used for E[EPS_t] and σ_t contains only quarters filed
    before the announcement date.
  * An 8-K event is mapped to the latest fiscal quarter that ended before
    it (within 120 days). EPS_t itself comes from the later 10-Q/10-K, which
    is fine because the 8-K earnings release made that number public on the
    announcement date.

Output: data/signals/sue_panel.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .earnings_dates import get_cik_map, sec_get

log = logging.getLogger(__name__)

QUARTER_DAYS = (70, 110)
ANNUAL_DAYS = (340, 390)
YEAR_TOL_DAYS = 20        # tolerance when matching "same quarter last year"
MAX_MAP_LAG_DAYS = 120    # 8-K must come within 120 days of the quarter end
MAX_FILE_LAG_DAYS = 100   # 10-Q/10-K must follow the release within 100 days
MIN_DIFFS_SRW = 4         # seasonal differences needed for σ (SRW)
MIN_PAIRS_AR1 = 8         # consecutive (ΔEPS_{t−1}, ΔEPS_t) pairs for AR(1)
STD_FLOOR = 0.01


# ── Quarterly EPS from XBRL (pure function; unit-tested) ─────────────────────

def quarterly_eps_from_facts(facts: dict) -> pd.DataFrame:
    """
    Build a clean quarterly EPS table from an XBRL companyfacts payload.
    Returns columns: period_start, period_end, eps, filed_date, derived_q4.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    rows = []
    for concept in ("EarningsPerShareDiluted", "EarningsPerShareBasic"):
        rows = gaap.get(concept, {}).get("units", {}).get("USD/shares", [])
        if rows:
            break
    cols = ["period_start", "period_end", "eps", "filed_date", "derived_q4"]
    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    if "start" not in df.columns:
        return pd.DataFrame(columns=cols)
    df = df[df["form"].isin(["10-Q", "10-K", "10-Q/A", "10-K/A"])].copy()
    df["period_start"] = pd.to_datetime(df["start"], errors="coerce")
    df["period_end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed_date"] = pd.to_datetime(df["filed"], errors="coerce")
    df["eps"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna(subset=["period_start", "period_end", "filed_date", "eps"])
    df["days"] = (df["period_end"] - df["period_start"]).dt.days

    def first_filed(x):
        return (x.sort_values("filed_date")
                 .drop_duplicates(subset=["period_start", "period_end"], keep="first"))

    q = first_filed(df[df["days"].between(*QUARTER_DAYS)])
    q = q.sort_values("filed_date").drop_duplicates("period_end", keep="first")
    fy = first_filed(df[df["days"].between(*ANNUAL_DAYS)])
    fy = fy.sort_values("filed_date").drop_duplicates("period_end", keep="first")

    # Derive Q4 = FY − (Q1 + Q2 + Q3) where Q4 is not reported directly
    derived = []
    for _, a in fy.iterrows():
        if (abs(q["period_end"] - a["period_end"]).dt.days <= 10).any():
            continue  # a 3-month Q4 fact already exists
        inside = q[(q["period_end"] > a["period_start"])
                   & (q["period_end"] < a["period_end"] - pd.Timedelta(days=30))]
        if len(inside) != 3:
            continue
        derived.append({
            "period_start": inside["period_end"].max() + pd.Timedelta(days=1),
            "period_end": a["period_end"],
            "eps": a["eps"] - inside["eps"].sum(),
            "filed_date": a["filed_date"],
            "derived_q4": True,
        })

    q = q.assign(derived_q4=False)[cols]
    out = pd.concat([q, pd.DataFrame(derived, columns=cols)], ignore_index=True)
    out["derived_q4"] = out["derived_q4"].astype(bool)
    return out.sort_values("period_end").reset_index(drop=True)


# ── Event → fiscal quarter mapping ───────────────────────────────────────────

def map_events_to_quarters(events: pd.DataFrame, eps: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the fiscal quarter (and its EPS row) to each event.
      8-K events       : latest quarter end in [announce − 120d, announce).
      10-Q/K fallbacks : quarter end within ±10 days of the report period.
    When several 8-Ks map to one quarter the earliest is kept.
    """
    if events.empty or eps.empty:
        return events.iloc[0:0].assign(period_end=pd.NaT)
    ends = pd.to_datetime(eps["period_end"]).sort_values().values
    out = []
    for _, ev in events.iterrows():
        a = ev["announce_date"]
        if ev["source"] == "10Q_fallback" and pd.notna(ev["fiscal_period_end"]):
            d = np.abs((ends - ev["fiscal_period_end"].to_datetime64())
                       .astype("timedelta64[D]").astype(int))
            if d.min() > 10:
                continue
            pe = ends[d.argmin()]
        else:
            cand = ends[(ends < a.to_datetime64())
                        & (ends >= (a - pd.Timedelta(days=MAX_MAP_LAG_DAYS)).to_datetime64())]
            if len(cand) == 0:
                continue
            pe = cand.max()
        out.append({**ev.to_dict(), "period_end": pd.Timestamp(pe)})
    if not out:
        return events.iloc[0:0].assign(period_end=pd.NaT)
    m = pd.DataFrame(out).sort_values("announce_date")
    n_before = len(m)
    m = m.drop_duplicates(subset="period_end", keep="first")
    if len(m) < n_before:
        log.debug(f"{n_before - len(m)} extra events mapped to an already-used quarter")
    return m.reset_index(drop=True)


# ── Expectation models (pure functions; unit-tested) ─────────────────────────

def _year_ago(ends: np.ndarray, target: pd.Timestamp) -> int | None:
    """Index of the quarter ending ≈ 1 year before `target`, else None."""
    if len(ends) == 0:
        return None
    gap = (target - pd.to_datetime(ends)).days.values
    ok = np.where(np.abs(gap - 365) <= YEAR_TOL_DAYS)[0]
    return int(ok[np.argmin(np.abs(gap[ok] - 365))]) if len(ok) else None


def _seasonal_diffs(hist: pd.DataFrame) -> pd.Series:
    """ΔEPS for each quarter in `hist` that has a year-ago quarter, by period_end."""
    ends = hist["period_end"].values
    eps = hist["eps"].values
    d = {}
    for i in range(len(hist)):
        j = _year_ago(ends[:i], pd.Timestamp(ends[i]))
        if j is not None:
            d[pd.Timestamp(ends[i])] = eps[i] - eps[j]
    return pd.Series(d, dtype=float)


def expected_eps(eps: pd.DataFrame, period_end: pd.Timestamp,
                 announce_date: pd.Timestamp, use_ar1: bool = True) -> dict:
    """E[EPS] and σ for quarter `period_end`, using only data filed before announce_date."""
    nan = {"eps_expected": np.nan, "std_error": np.nan, "model": "insufficient_data",
           "n_quarters": 0}
    hist = (eps[(eps["filed_date"] < announce_date) & (eps["period_end"] < period_end)]
            .sort_values("period_end").reset_index(drop=True))
    nan["n_quarters"] = len(hist)
    j = _year_ago(hist["period_end"].values, period_end)
    if j is None:
        return nan
    eps_lag4 = float(hist.loc[j, "eps"])
    diffs = _seasonal_diffs(hist)

    if use_ar1 and len(diffs) >= MIN_PAIRS_AR1 + 1:
        # consecutive-quarter pairs only (gaps break the AR(1) recursion)
        idx = diffs.index
        gaps = np.diff(idx.values).astype("timedelta64[D]").astype(int)
        consecutive = (gaps >= 70) & (gaps <= 110)
        x = diffs.values[:-1][consecutive]
        y = diffs.values[1:][consecutive]
        last_gap = (period_end - idx[-1]).days
        if len(x) >= MIN_PAIRS_AR1 and 70 <= last_gap <= 110:
            xd = x - x.mean()
            if xd @ xd > 1e-10:
                beta = float(np.clip((xd @ y) / (xd @ xd), -0.99, 0.99))
                alpha = float(y.mean() - beta * x.mean())
                resid = y - (alpha + beta * x)
                std = max(float(np.std(resid, ddof=2)), STD_FLOOR)
                return {"eps_expected": eps_lag4 + alpha + beta * diffs.values[-1],
                        "std_error": std, "model": "ar1", "n_quarters": len(hist)}

    if len(diffs) < MIN_DIFFS_SRW:
        return nan
    std = max(float(np.std(diffs.values[-8:], ddof=1)), STD_FLOOR)
    return {"eps_expected": eps_lag4, "std_error": std, "model": "seasonal_rw",
            "n_quarters": len(hist)}


# ── Per-ticker and panel builders ────────────────────────────────────────────

def compute_sue_for_ticker(ticker: str, events: pd.DataFrame, eps: pd.DataFrame,
                           use_ar1: bool = True) -> pd.DataFrame:
    events = events.copy()
    events["announce_date"] = pd.to_datetime(events["announce_date"])
    events["fiscal_period_end"] = pd.to_datetime(events["fiscal_period_end"])
    mapped = map_events_to_quarters(events, eps)
    rows = []
    for _, ev in mapped.iterrows():
        cur = eps[eps["period_end"] == ev["period_end"]].iloc[0]
        if (cur["filed_date"] - ev["announce_date"]).days > MAX_FILE_LAG_DAYS:
            continue  # the matched EPS was filed implausibly late: mismatch
        m = expected_eps(eps, ev["period_end"], ev["announce_date"], use_ar1)
        sue = ((cur["eps"] - m["eps_expected"]) / m["std_error"]
               if pd.notna(m["eps_expected"]) else np.nan)
        rows.append({
            "ticker": ticker,
            "announce_date": ev["announce_date"],
            "fiscal_period_end": ev["period_end"],
            "eps_actual": cur["eps"],
            "eps_expected": m["eps_expected"],
            "eps_surprise": cur["eps"] - m["eps_expected"],
            "std_error": m["std_error"],
            "sue": sue,
            "sue_model": m["model"],
            "n_quarters_used": m["n_quarters"],
            "derived_q4": bool(cur["derived_q4"]),
            "source": ev["source"],
        })
    return pd.DataFrame(rows)


def fetch_eps(cik: str) -> pd.DataFrame:
    facts = sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                    cache_key=f"companyfacts_{cik}")
    return quarterly_eps_from_facts(facts) if facts else pd.DataFrame()


def build_sue_panel(earnings_path: Path = config.EARNINGS_CSV,
                    output_path: Path = config.SUE_CSV,
                    use_ar1: bool = True, min_obs: int = 3) -> pd.DataFrame:
    if not earnings_path.exists():
        raise FileNotFoundError(f"{earnings_path} not found; run step 1 first")
    config.ensure_dirs()
    ann = pd.read_csv(earnings_path, parse_dates=["announce_date", "fiscal_period_end"])
    tickers = ann["ticker"].unique().tolist()
    cik_map = get_cik_map(tickers)
    log.info(f"Computing SUE for {len(tickers)} tickers, {len(ann)} events "
             f"({'AR(1) with SRW fallback' if use_ar1 else 'SRW only'})")

    frames = []
    for i, t in enumerate(tickers, 1):
        log.info(f"[{i}/{len(tickers)}] {t}")
        cik = cik_map.get(t.upper())
        eps = fetch_eps(cik) if cik else pd.DataFrame()
        if len(eps) < 5:
            log.warning(f"{t}: insufficient XBRL EPS history")
            continue
        res = compute_sue_for_ticker(t, ann[ann["ticker"] == t], eps, use_ar1)
        if len(res) >= min_obs:
            frames.append(res)

    if not frames:
        log.error("No SUE computed; check XBRL availability")
        return pd.DataFrame()
    panel = (pd.concat(frames, ignore_index=True)
               .sort_values(["ticker", "announce_date"]).reset_index(drop=True))

    # Winsorize at the 1st/99th percentile. The signal is only used through
    # cross-sectional ranks, so this just stops outliers dominating Pearson
    # statistics; it does not change quintile membership.
    lo, hi = panel["sue"].quantile([0.01, 0.99])
    panel["sue_winsor"] = panel["sue"].clip(lo, hi)

    n = len(panel)
    log.info(f"SUE panel: {n} events, {panel['ticker'].nunique()} tickers")
    log.info(f"  SUE available: {panel['sue'].notna().mean():.1%}  "
             f"AR(1): {(panel['sue_model'] == 'ar1').mean():.1%}  "
             f"SRW: {(panel['sue_model'] == 'seasonal_rw').mean():.1%}  "
             f"derived Q4: {panel['derived_q4'].mean():.1%}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    log.info(f"Saved to {output_path}")
    return panel


def main():
    config.setup_logging()
    p = argparse.ArgumentParser(description="Compute time-series SUE")
    p.add_argument("--earnings-csv", default=str(config.EARNINGS_CSV))
    p.add_argument("--output", default=str(config.SUE_CSV))
    p.add_argument("--no-ar1", action="store_true", help="Seasonal RW only")
    a = p.parse_args()
    build_sue_panel(Path(a.earnings_csv), Path(a.output), use_ar1=not a.no_ar1)


if __name__ == "__main__":
    main()
