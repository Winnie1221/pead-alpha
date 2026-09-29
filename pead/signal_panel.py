"""
Step 3: merge SUE with prices into the event-level signal panel.

Timing convention (see config.ENTRY_LAG):
    day 0     = first trading day on or after the 8-K filing date
    entry     = close of day 0 + ENTRY_LAG   (the first close after the news
                that we could actually trade; most releases come after the bell)
    ret_Nd    = close(entry + N) / close(entry) − 1
    excess    = ret_Nd − SPY return over the same window

The announcement reaction itself, close(day −1) → close(entry), is kept
separately as `ann_ret` / `ann_excess_ret`. It is NOT part of the drift
returns, since it is realized before a trader could act on the SUE.

Output: data/signals/signal_panel.parquet (+ .csv)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config

log = logging.getLogger(__name__)

SOURCE_RANK = {"8K_item202": 0, "10Q_fallback": 1}
MERGE_WINDOW_DAYS = 3


def download_prices(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Adjusted daily closes, wide (date × ticker)."""
    import yfinance as yf
    log.info(f"Downloading prices for {len(tickers)} tickers ({start_date} → {end_date})")
    px = yf.download(tickers, start=start_date, end=end_date,
                     auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px.index = pd.to_datetime(px.index)
    return px.sort_index()


def deduplicate_events(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, date); events within 3 days collapse to the better source."""
    df = df.copy()
    df["source_rank"] = df["source"].map(SOURCE_RANK).fillna(99).astype(int)
    df = (df.sort_values(["ticker", "announce_date", "source_rank"])
            .drop_duplicates(subset=["ticker", "announce_date"], keep="first"))
    out = []
    for _, g in df.groupby("ticker"):
        g = g.sort_values("announce_date").reset_index(drop=True)
        keep = np.ones(len(g), dtype=bool)
        for i in range(1, len(g)):
            if (g.loc[i, "announce_date"] - g.loc[i - 1, "announce_date"]).days <= MERGE_WINDOW_DAYS:
                worse = i if g.loc[i, "source_rank"] >= g.loc[i - 1, "source_rank"] else i - 1
                keep[worse] = False
        out.append(g[keep])
    res = pd.concat(out, ignore_index=True) if out else df.iloc[0:0]
    log.info(f"  Deduplication: {len(df)} → {len(res)} events")
    return res


def compute_forward_returns(prices: pd.DataFrame, events: pd.DataFrame,
                            horizons: list[int] = config.HORIZONS,
                            entry_lag: int = config.ENTRY_LAG,
                            market: str = config.MARKET_TICKER) -> pd.DataFrame:
    """Forward, market-excess and announcement returns for each event row."""
    days = prices.index
    day0 = days.searchsorted(pd.to_datetime(events["announce_date"]).values)
    entry = day0 + entry_lag
    P = prices
    mkt = P[market] if market in P.columns else None

    def window_ret(series, i0, i1):
        out = np.full(len(i0), np.nan)
        ok = (i0 >= 0) & (i1 < len(days)) & (i0 < len(days))
        v0 = series.values[i0[ok]]
        v1 = series.values[i1[ok]]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[ok] = np.where(v0 > 0, v1 / v0 - 1, np.nan)
        return out

    res = {}
    tickers = events["ticker"].values
    for h in horizons:
        r = np.full(len(events), np.nan)
        for t in np.unique(tickers):
            if t not in P.columns:
                continue
            m = tickers == t
            r[m] = window_ret(P[t], entry[m], entry[m] + h)
        m_r = window_ret(mkt, entry, entry + h) if mkt is not None else np.full(len(events), np.nan)
        res[f"ret_{h}d"] = r
        res[f"mkt_ret_{h}d"] = m_r
        res[f"excess_ret_{h}d"] = r - m_r

    a = np.full(len(events), np.nan)
    for t in np.unique(tickers):
        if t not in P.columns:
            continue
        m = tickers == t
        a[m] = window_ret(P[t], day0[m] - 1, entry[m])
    m_a = window_ret(mkt, day0 - 1, entry) if mkt is not None else np.full(len(events), np.nan)
    res["ann_ret"] = a
    res["ann_excess_ret"] = a - m_a
    return pd.DataFrame(res, index=events.index)


def build_signal_panel(sue_path: Path = config.SUE_CSV,
                       output_pq: Path = config.SIGNAL_PQ,
                       start_date: str = config.DEFAULT_START,
                       end_date: str = config.DEFAULT_END) -> pd.DataFrame:
    if not sue_path.exists():
        raise FileNotFoundError(f"{sue_path} not found; run step 2 first")
    sue = pd.read_csv(sue_path, parse_dates=["announce_date", "fiscal_period_end"])
    log.info(f"Loaded SUE panel: {len(sue)} rows, {sue['ticker'].nunique()} tickers")
    sue = deduplicate_events(sue).reset_index(drop=True)

    tickers = sorted(sue["ticker"].unique().tolist())
    prices = download_prices(tickers + [config.MARKET_TICKER], start_date, end_date)
    panel = pd.concat([sue, compute_forward_returns(prices, sue)], axis=1)

    log.info(f"Signal panel: {len(panel)} events, {panel['ticker'].nunique()} tickers, "
             f"{panel['announce_date'].min().date()} → {panel['announce_date'].max().date()}")
    for h in config.HORIZONS:
        sub = panel[["sue_winsor", f"excess_ret_{h}d"]].dropna()
        if len(sub) > 10:
            ic, p = spearmanr(sub.iloc[:, 0], sub.iloc[:, 1])
            log.info(f"  pooled IC {h:2d}d: {ic:+.3f} (p={p:.3f}, n={len(sub)})")

    output_pq.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_pq, index=False)
    panel.to_csv(output_pq.with_suffix(".csv"), index=False)
    log.info(f"Saved to {output_pq}")
    return panel


def main():
    config.setup_logging()
    p = argparse.ArgumentParser(description="Build the PEAD signal panel")
    p.add_argument("--sue-csv", default=str(config.SUE_CSV))
    p.add_argument("--output", default=str(config.SIGNAL_PQ))
    p.add_argument("--start-date", default=config.DEFAULT_START)
    p.add_argument("--end-date", default=config.DEFAULT_END)
    a = p.parse_args()
    build_signal_panel(Path(a.sue_csv), Path(a.output), a.start_date, a.end_date)


if __name__ == "__main__":
    main()
