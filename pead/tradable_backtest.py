"""
Step 5: tradable implementation — daily-rebalanced, dollar-neutral quintile book.

Construction (point-in-time):
  * An event becomes active at the close of day 0 + ENTRY_LAG (the first
    tradable close after the 8-K; same convention as the event study) and
    stays active for HOLD_DAYS trading days.
  * Each day, rank active names by SUE; long the top quintile, short the
    bottom quintile, equal weight within each leg, 50% gross per leg.
  * Fewer than MIN_NAMES active names → flat (cash).
  * Weights set at close t earn the close-to-close return of day t+1.

Turnover (one-way, drift-adjusted):
    turnover_t = ½ Σ_i |w_{i,t} − w̃_{i,t}|,  w̃_{i,t} = w_{i,t−1}(1 + r_{i,t}) / (1 + r_{p,t})
Costs:
    net_t = gross_t − turnover_t × cost_bps × 1e−4   (5 and 10 bps one-way)
Breakeven cost = mean(gross) / mean(turnover), in bps.

Output: results/tradable_backtest.json, results/tradable_daily_returns.csv
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .event_study import load_panel
from .signal_panel import download_prices

log = logging.getLogger(__name__)


def build_active_signal_matrix(panel: pd.DataFrame, trading_days: pd.DatetimeIndex,
                               hold_days: int = config.HOLD_DAYS,
                               entry_lag: int = config.ENTRY_LAG) -> pd.DataFrame:
    """(date × ticker) SUE while an event is inside its holding window, else NaN."""
    ev = (panel[["ticker", "announce_date", "sue_winsor"]].dropna()
          .sort_values(["ticker", "announce_date"])
          .drop_duplicates(["ticker", "announce_date"], keep="last"))
    ev["start"] = trading_days.searchsorted(pd.to_datetime(ev["announce_date"]).values) + entry_lag
    ev = ev[ev["start"] < len(trading_days)]

    days = np.arange(len(trading_days))
    active = {}
    for t, g in ev.groupby("ticker"):
        start, sue = g["start"].values, g["sue_winsor"].values
        order = np.argsort(start, kind="stable")
        start, sue = start[order], sue[order]
        loc = np.searchsorted(start, days, side="right") - 1  # latest started event
        col = np.full(len(days), np.nan)
        has = loc >= 0
        age = np.where(has, days - start[np.clip(loc, 0, None)], -1)
        use = has & (age < hold_days)
        col[use] = sue[loc[use]]
        active[t] = col
    out = pd.DataFrame(active, index=trading_days)
    log.info(f"  Active-signal matrix {out.shape}; "
             f"{int((out.notna().sum(axis=1) > 0).sum())} days with ≥1 active name")
    return out


def quintile_weights(active: pd.DataFrame, min_names: int = config.MIN_NAMES) -> pd.DataFrame:
    """Long top / short bottom SUE quintile, 50% gross per leg."""
    w = pd.DataFrame(0.0, index=active.index, columns=active.columns)
    counts = active.notna().sum(axis=1)
    for day in active.index[counts >= min_names]:
        row = active.loc[day].dropna()
        pct = row.rank(pct=True)
        longs, shorts = row.index[pct > 0.8], row.index[pct <= 0.2]
        if len(longs) and len(shorts):
            w.loc[day, longs] = 0.5 / len(longs)
            w.loc[day, shorts] = -0.5 / len(shorts)
    n = int((w.abs().sum(axis=1) > 0).sum())
    log.info(f"  Invested on {n}/{len(w)} days ({n/len(w):.1%})")
    return w


def portfolio_returns(weights: pd.DataFrame, prices: pd.DataFrame,
                      costs_bps: list[float] = config.COST_SCENARIOS_BPS) -> pd.DataFrame:
    """Gross return, drift-adjusted turnover and net returns per day."""
    rets = prices.reindex(columns=weights.columns).pct_change(fill_method=None)
    r = rets.reindex(weights.index).fillna(0.0).values
    w = weights.values
    w_prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    gross = np.einsum("ij,ij->i", w_prev, r)

    turnover = np.empty(len(w))
    for t in range(len(w)):
        denom = 1 + gross[t]
        drifted = w_prev[t] * (1 + r[t]) / denom if abs(denom) > 1e-12 else w_prev[t]
        turnover[t] = 0.5 * np.abs(w[t] - drifted).sum()

    out = pd.DataFrame({"gross_ret": gross, "turnover": turnover}, index=weights.index)
    for c in costs_bps:
        out[f"net_ret_{int(c)}bps"] = out["gross_ret"] - out["turnover"] * c * 1e-4
    out.index.name = "date"
    return out


def summarize(ret: pd.Series, turnover: pd.Series) -> dict:
    r = ret.dropna().values
    if len(r) < 20:
        return {"n_days": int(len(r))}
    ann_ret = r.mean() * config.TRADING_DAYS
    ann_vol = r.std(ddof=1) * np.sqrt(config.TRADING_DAYS)
    cum = np.cumprod(1 + r)
    mdd = float(((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)).min())
    t = turnover.reindex(ret.index).fillna(0.0).values
    return {
        "n_days": int(len(r)),
        "ann_return": round(float(ann_ret), 4),
        "ann_vol": round(float(ann_vol), 4),
        "sharpe": round(float(ann_ret / ann_vol), 3) if ann_vol > 0 else None,
        "max_drawdown": round(mdd, 4),
        "ann_turnover_x": round(float(t.mean() * config.TRADING_DAYS), 2),
        "breakeven_cost_bps": round(float(r.mean() / t.mean() * 1e4), 1) if t.mean() > 0 else None,
    }


def run_tradable_backtest(signal_path: Path = config.SIGNAL_PQ,
                          output_dir: Path = config.RESULTS_DIR,
                          start_date: str = "2008-01-01",
                          end_date: str = config.DEFAULT_END,
                          prices: pd.DataFrame | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel(signal_path)
    log.info(f"Loaded signal panel: {len(panel)} events, {panel['ticker'].nunique()} tickers")
    tickers = sorted(panel["ticker"].unique())
    if prices is None:
        prices = download_prices(tickers, start_date, end_date)
    prices = prices.reindex(columns=tickers)

    active = build_active_signal_matrix(panel, prices.index)
    weights = quintile_weights(active)
    acct = portfolio_returns(weights, prices)
    acct.to_csv(output_dir / "tradable_daily_returns.csv")

    res = {"entry_lag_days": config.ENTRY_LAG, "hold_days": config.HOLD_DAYS,
           "gross": summarize(acct["gross_ret"], acct["turnover"])}
    for c in config.COST_SCENARIOS_BPS:
        k = f"net_ret_{int(c)}bps"
        res[k] = summarize(acct[k], acct["turnover"])
    res["subperiods"] = {}
    for name, (s, e) in config.SUBPERIODS.items():
        sub = acct.loc[s:e]
        if len(sub) >= 60:
            res["subperiods"][name] = {
                "gross": summarize(sub["gross_ret"], sub["turnover"]),
                "net_10bps": summarize(sub["net_ret_10bps"], sub["turnover"]),
            }

    log.info("=" * 60 + "\nTRADABLE BACKTEST (daily rebalanced, dollar neutral)\n" + "=" * 60)
    for k in ["gross"] + [f"net_ret_{int(c)}bps" for c in config.COST_SCENARIOS_BPS]:
        s = res[k]
        log.info(f"  {k:14s} Sharpe={s['sharpe']:.2f}  Ret={s['ann_return']:.2%}  "
                 f"DD={s['max_drawdown']:.2%}  Turn={s['ann_turnover_x']:.1f}x/yr  "
                 f"Breakeven={s['breakeven_cost_bps']}bps")
    log.info("  Subperiod gross Sharpe: " + ", ".join(
        f"{k}={v['gross']['sharpe']}" for k, v in res["subperiods"].items()))

    (output_dir / "tradable_backtest.json").write_text(json.dumps(res, indent=2, default=str))
    log.info(f"Saved to {output_dir / 'tradable_backtest.json'}")
    return res


def main():
    config.setup_logging()
    p = argparse.ArgumentParser(description="Tradable PEAD backtest with turnover and costs")
    p.add_argument("--signal-path", default=str(config.SIGNAL_PQ))
    p.add_argument("--output-dir", default=str(config.RESULTS_DIR))
    p.add_argument("--start-date", default="2008-01-01")
    p.add_argument("--end-date", default=config.DEFAULT_END)
    a = p.parse_args()
    run_tradable_backtest(Path(a.signal_path), Path(a.output_dir), a.start_date, a.end_date)


if __name__ == "__main__":
    main()
