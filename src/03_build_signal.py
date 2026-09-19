"""
03_build_signal.py
==================
Module 3 of the PEAD Alpha Pipeline.

Purpose:
    1. Download daily price data (yfinance) for the ticker universe.
    2. Merge SUE panel with prices to compute post-announcement returns.
    3. Compute market-adjusted (excess) returns at horizons: +1, +5, +21, +63 days.
    4. Deduplicate events: keep best-source row per (ticker, announce_date).
    5. Output the final signal panel for IC analysis and backtesting.

Key design choices:
    - Market-adjusted returns: raw return minus SPY return over same window.
      This controls for market beta and is standard in academic PEAD literature.
    - Announcement-day return (day 0) is excluded from forward returns.
      We assume execution on OPEN of day+1 (avoids close-of-day timing ambiguity).
    - No look-ahead: forward returns are computed from day t+1 open perspective,
      using close-to-close returns starting from the close on announcement day.
    - SUE is winsorized at 1%/99% (done in Module 2); used as-is here.
    - Event deduplication: (ticker, announce_date) → keep 8K_item202 > 8K_no_item
      > 10Q_fallback. Multiple events within 3 calendar days are merged.

Output:
    data/signals/signal_panel.parquet
    Columns: ticker, announce_date, sue, sue_winsor, sue_model,
             ret_1d, ret_5d, ret_21d, ret_63d,
             mkt_ret_1d, ..., excess_ret_1d, ...,
             source, n_quarters_used

Author: Tingxuan Wu
"""

import warnings
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).parent.parent
DATA_DIR    = _ROOT / "data"
SIGNALS_DIR = DATA_DIR / "signals"
SUE_CSV     = SIGNALS_DIR / "sue_panel.csv"
OUTPUT_PQ   = SIGNALS_DIR / "signal_panel.parquet"
OUTPUT_CSV  = SIGNALS_DIR / "signal_panel.csv"

SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
HORIZONS      = [1, 5, 21, 63]       # trading days
MARKET_TICKER = "SPY"                # market benchmark
SOURCE_RANK   = {                    # lower = better PIT quality
    "8K_item202":  0,
    "8K_no_item":  1,
    "10Q_fallback": 2,
}
MERGE_WINDOW_DAYS = 3                # collapse events within 3 calendar days


# ── Step 1: Download prices ────────────────────────────────────────────────────

def download_prices(
    tickers: list,
    start_date: str = "2005-01-01",
    end_date: str   = "2026-09-30",
) -> pd.DataFrame:
    """
    Download adjusted daily close prices for tickers + SPY.
    Returns wide DataFrame: index=date, columns=tickers.
    """
    universe = list(set(tickers + [MARKET_TICKER]))
    log.info(f"Downloading prices for {len(universe)} tickers ({start_date} → {end_date})")

    prices = yf.download(
        universe,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )["Close"]

    # yfinance may return MultiIndex or flat columns
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    n_missing = prices.isna().mean()
    for t in universe:
        if t in n_missing and n_missing[t] > 0.3:
            log.warning(f"  {t}: {n_missing[t]:.0%} missing prices")

    log.info(f"  Prices downloaded: {prices.shape} ({prices.index[0].date()} → {prices.index[-1].date()})")
    return prices


# ── Step 2: Compute forward returns ───────────────────────────────────────────

def compute_forward_returns(
    prices: pd.DataFrame,
    announce_dates: pd.DatetimeIndex,
    ticker_col: pd.Series,
    horizons: list = HORIZONS,
) -> pd.DataFrame:
    """
    For each (ticker, announce_date), compute:
        ret_Nd    = cumulative return from close(day 0) to close(day+N)
        mkt_ret_Nd = same for SPY
        excess_ret_Nd = ret_Nd - mkt_ret_Nd

    Day 0 = announcement date (or next trading day if it's a non-trading day).
    Returns computed from close(day 0) → close(day 0+N), giving N-day hold.
    """
    trading_days = prices.index
    log.info(f"Computing forward returns for {len(announce_dates)} events at horizons {horizons}")

    results = []
    for ann_date, ticker in zip(announce_dates, ticker_col):
        if ticker not in prices.columns:
            continue

        # Find day 0: announcement date or next trading day
        future = trading_days[trading_days >= ann_date]
        if len(future) == 0:
            continue
        day0_idx = trading_days.get_loc(future[0])

        row = {"_day0_idx": day0_idx}
        for h in horizons:
            day_h_idx = day0_idx + h
            if day_h_idx >= len(trading_days):
                row[f"ret_{h}d"]        = np.nan
                row[f"mkt_ret_{h}d"]    = np.nan
                row[f"excess_ret_{h}d"] = np.nan
                continue

            p0_stock = prices[ticker].iloc[day0_idx]
            ph_stock = prices[ticker].iloc[day_h_idx]
            p0_spy   = prices[MARKET_TICKER].iloc[day0_idx] if MARKET_TICKER in prices.columns else np.nan
            ph_spy   = prices[MARKET_TICKER].iloc[day_h_idx] if MARKET_TICKER in prices.columns else np.nan

            ret_stock = ph_stock / p0_stock - 1 if p0_stock > 0 else np.nan
            ret_mkt   = ph_spy / p0_spy - 1 if (not np.isnan(p0_spy) and p0_spy > 0) else np.nan

            row[f"ret_{h}d"]        = ret_stock
            row[f"mkt_ret_{h}d"]    = ret_mkt
            row[f"excess_ret_{h}d"] = ret_stock - ret_mkt if not np.isnan(ret_mkt) else ret_stock

        results.append(row)

    return pd.DataFrame(results).drop(columns=["_day0_idx"], errors="ignore")


# ── Step 3: Deduplicate events ─────────────────────────────────────────────────

def deduplicate_events(sue_df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle duplicate (ticker, announce_date) rows:
    1. Assign source quality rank (0=best).
    2. For each (ticker, announce_date), keep best-source row.
    3. Merge events within MERGE_WINDOW_DAYS calendar days of each other
       (e.g., 8-K filed same day as 10-Q gets collapsed).

    Returns deduplicated DataFrame.
    """
    df = sue_df.copy()
    df["source_rank"] = df["source"].map(SOURCE_RANK).fillna(99).astype(int)

    # Step 1: keep best source per (ticker, announce_date)
    df = (df.sort_values(["ticker", "announce_date", "source_rank"])
            .drop_duplicates(subset=["ticker", "announce_date"], keep="first"))

    # Step 2: merge events within MERGE_WINDOW_DAYS per ticker
    deduped = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("announce_date").reset_index(drop=True)
        keep = [True] * len(grp)
        for i in range(1, len(grp)):
            gap = (grp.loc[i, "announce_date"] - grp.loc[i-1, "announce_date"]).days
            if gap <= MERGE_WINDOW_DAYS:
                # Keep the one with better source_rank (lower = better)
                if grp.loc[i, "source_rank"] >= grp.loc[i-1, "source_rank"]:
                    keep[i] = False
                else:
                    keep[i-1] = False
        deduped.append(grp[keep])

    result = pd.concat(deduped, ignore_index=True)
    log.info(f"  Deduplication: {len(sue_df)} → {len(result)} events")
    return result


# ── Step 4: Main pipeline ──────────────────────────────────────────────────────

def build_signal_panel(
    sue_path: Path   = SUE_CSV,
    output_pq: Path  = OUTPUT_PQ,
    output_csv: Path = OUTPUT_CSV,
    start_date: str  = "2005-01-01",
    end_date: str    = "2026-09-30",
) -> pd.DataFrame:
    """
    Main entry point. Produces the final signal panel for backtesting.
    """
    if not sue_path.exists():
        raise FileNotFoundError(f"sue_panel.csv not found at {sue_path}. Run Module 2 first.")

    # Load SUE panel
    log.info(f"Loading SUE panel from {sue_path}")
    sue = pd.read_csv(sue_path, parse_dates=["announce_date", "fiscal_period_end"])
    log.info(f"  Raw SUE rows: {len(sue)}, tickers: {sue['ticker'].nunique()}")

    # Deduplicate
    sue = deduplicate_events(sue)

    # Get ticker universe
    tickers = sue["ticker"].unique().tolist()

    # Download prices
    prices = download_prices(tickers, start_date=start_date, end_date=end_date)

    # Compute forward returns
    fwd = compute_forward_returns(
        prices        = prices,
        announce_dates= sue["announce_date"].values,
        ticker_col    = sue["ticker"].values,
        horizons      = HORIZONS,
    )

    # Merge with SUE
    panel = pd.concat([sue.reset_index(drop=True), fwd], axis=1)

    # ── Quality report ────────────────────────────────────────────────────────
    n = len(panel)
    log.info(f"\n{'='*55}")
    log.info(f"Signal panel: {n} events, {panel['ticker'].nunique()} tickers")
    log.info(f"  Date range: {panel['announce_date'].min().date()} → {panel['announce_date'].max().date()}")
    log.info(f"  SUE coverage:   {panel['sue'].notna().mean():.1%}")
    for h in HORIZONS:
        cov = panel[f"excess_ret_{h}d"].notna().mean()
        log.info(f"  excess_ret_{h:2d}d: {cov:.1%} coverage")

    # Quick IC preview (Spearman rank correlation: SUE vs excess return)
    from scipy.stats import spearmanr
    log.info(f"\n  Quick IC preview (Spearman, pooled):")
    for h in HORIZONS:
        sub = panel[["sue_winsor", f"excess_ret_{h}d"]].dropna()
        if len(sub) > 10:
            ic, pval = spearmanr(sub["sue_winsor"], sub[f"excess_ret_{h}d"])
            log.info(f"    {h:2d}d horizon:  IC={ic:+.3f}  p={pval:.3f}  n={len(sub)}")

    log.info(f"{'='*55}")

    # Save
    panel.to_parquet(output_pq, index=False)
    panel.to_csv(output_csv, index=False)
    log.info(f"Saved to {output_pq}")

    return panel


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from scipy.stats import spearmanr

    parser = argparse.ArgumentParser(description="Build PEAD signal panel")
    parser.add_argument("--sue-csv",    default=str(SUE_CSV))
    parser.add_argument("--output-pq",  default=str(OUTPUT_PQ))
    parser.add_argument("--output-csv", default=str(OUTPUT_CSV))
    parser.add_argument("--start-date", default="2005-01-01")
    parser.add_argument("--end-date",   default="2026-09-30")
    args = parser.parse_args()

    panel = build_signal_panel(
        sue_path   = Path(args.sue_csv),
        output_pq  = Path(args.output_pq),
        output_csv = Path(args.output_csv),
        start_date = args.start_date,
        end_date   = args.end_date,
    )

    if not panel.empty:
        print("\nSample signal panel:")
        cols = ["ticker", "announce_date", "sue_winsor",
                "excess_ret_1d", "excess_ret_5d", "excess_ret_21d", "excess_ret_63d"]
        print(panel[cols].dropna(subset=["excess_ret_5d"]).head(15).to_string(index=False))
