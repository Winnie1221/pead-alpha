"""
Step 4: statistical evaluation of the signal (SUE → post-entry excess returns).

  1. IC time series  — quarterly Spearman IC, ICIR, Newey-West t-stat, hit rate
  2. Alpha decay     — pooled IC at 1 / 5 / 21 / 63 trading days
  3. Event-time L/S  — per quarterly cohort, Q5 − Q1 mean excess return over the
                       holding horizon. Annualized with 4 cohorts per year, i.e.
                       the book is invested `horizon` days per quarter. The
                       always-invested daily book lives in tradable_backtest.py.
  4. VIX regimes     — IC split by VIX level on the announcement date
                       (stress ≥ 25, neutral 15–25, risk-on < 15)

Output: results/event_study.json, results/ic_series.csv, results/quintile_returns.csv
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config

log = logging.getLogger(__name__)

PERIODS_PER_YEAR = {"Q": 4, "M": 12}


def load_panel(path: Path = config.SIGNAL_PQ) -> pd.DataFrame:
    panel = (pd.read_parquet(path) if path.suffix == ".parquet" and path.exists()
             else pd.read_csv(path.with_suffix(".csv")))
    panel["announce_date"] = pd.to_datetime(panel["announce_date"])
    return panel


def newey_west_se(x: np.ndarray, lags: int = config.NW_LAGS) -> float:
    """HAC standard error of the mean with a Bartlett kernel."""
    n = len(x)
    xd = x - x.mean()
    var = xd @ xd / n
    for l in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - l / (lags + 1)) * (xd[l:] @ xd[:-l]) / n
    return float(np.sqrt(max(var, 1e-12) / n))


def compute_ic_series(panel: pd.DataFrame, horizon: int, freq: str = "Q") -> pd.DataFrame:
    col = f"excess_ret_{horizon}d"
    df = panel[["announce_date", "sue_winsor", col]].dropna().copy()
    df["period"] = df["announce_date"].dt.to_period(freq)
    rows = []
    for period, g in df.groupby("period"):
        if len(g) >= 10:
            rows.append({"period": str(period),
                         "ic": stats.spearmanr(g["sue_winsor"], g[col])[0],
                         "n_obs": len(g)})
    return pd.DataFrame(rows)


def compute_ic_stats(ic: pd.DataFrame) -> dict:
    x = ic["ic"].dropna().values
    if len(x) < 3:
        return {}
    se = newey_west_se(x)
    t = x.mean() / se
    return {
        "mean_ic": round(float(x.mean()), 4),
        "std_ic": round(float(x.std(ddof=1)), 4),
        "icir": round(float(x.mean() / x.std(ddof=1)), 3),
        "t_statistic_nw": round(float(t), 3),
        "p_value": round(float(2 * (1 - stats.t.cdf(abs(t), df=len(x) - 1))), 4),
        "ic_positive_pct": round(float((x > 0).mean()), 3),
        "n_periods": int(len(x)),
    }


def compute_alpha_decay(panel: pd.DataFrame) -> dict:
    out = {}
    for h in config.HORIZONS:
        df = panel[["sue_winsor", f"excess_ret_{h}d"]].dropna()
        if len(df) >= 10:
            ic, p = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
            out[f"{h}d"] = {"ic": round(float(ic), 4), "p_val": round(float(p), 4),
                            "n": int(len(df))}
    return out


def quintile_backtest(panel: pd.DataFrame, horizon: int, freq: str = "Q"):
    col = f"excess_ret_{horizon}d"
    df = panel[["announce_date", "sue_winsor", col]].dropna().copy()
    df["period"] = df["announce_date"].dt.to_period(freq)
    rows = []
    for period, g in df.groupby("period"):
        if len(g) < 10:
            continue
        q = pd.qcut(g["sue_winsor"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        means = g.groupby(q, observed=False)[col].mean()
        rows.append({"period": str(period), **{f"q{k}_return": means[k] for k in range(1, 6)},
                     "ls_return": means[5] - means[1], "n_events": len(g)})
    if not rows:
        return pd.DataFrame(), {}
    ls = pd.DataFrame(rows)
    x = ls["ls_return"].values
    ppy = PERIODS_PER_YEAR.get(freq, 4)
    ann_ret = x.mean() * ppy
    ann_vol = x.std(ddof=1) * np.sqrt(ppy)
    cum = np.cumprod(1 + x)
    mdd = float(((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)).min())
    return ls, {
        "horizon_days": horizon,
        "mean_period_ls_return": round(float(x.mean()), 4),
        "ann_return": round(float(ann_ret), 4),
        "ann_vol": round(float(ann_vol), 4),
        "sharpe_ratio": round(float(ann_ret / ann_vol), 3) if ann_vol > 0 else None,
        "max_drawdown": round(mdd, 4),
        "t_statistic_nw": round(float(x.mean() / newey_west_se(x)), 3),
        "hit_rate": round(float((x > 0).mean()), 3),
        "n_periods": int(len(x)),
    }


def regime_ic(panel: pd.DataFrame, horizon: int) -> dict:
    try:
        import yfinance as yf
        vix = yf.download(config.VIX_TICKER, start="2004-01-01", end=config.DEFAULT_END,
                          auto_adjust=True, progress=False)["Close"]
        if isinstance(vix, pd.DataFrame):
            vix = vix.iloc[:, 0]
        vix.index = pd.to_datetime(vix.index)
        vix = vix[~vix.index.duplicated(keep="last")].sort_index()
    except Exception as e:  # network or API change
        log.warning(f"VIX download failed: {e}")
        return {}
    col = f"excess_ret_{horizon}d"
    df = panel[["announce_date", "sue_winsor", col]].dropna().copy()
    # VIX close on the last trading day ≤ announcement date (known at entry)
    try:
        vix_filled = vix.reindex(vix.index.union(pd.to_datetime(df["announce_date"]))).ffill()
        vix_filled = vix_filled[~vix_filled.index.duplicated(keep="last")]
        df["vix"] = pd.to_datetime(df["announce_date"]).map(vix_filled).values
    except Exception as e:
        log.warning(f"VIX regime mapping failed, skipping regime IC: {e}")
        return {}
    df = df.dropna(subset=["vix"])
    groups = {"stress": df[df["vix"] >= 25],
              "neutral": df[(df["vix"] >= 15) & (df["vix"] < 25)],
              "risk_on": df[df["vix"] < 15]}
    out = {}
    for k, g in groups.items():
        if len(g) < 30:
            out[k] = {"ic": None, "n": int(len(g))}
            continue
        ic, p = stats.spearmanr(g["sue_winsor"], g[col])
        out[k] = {"ic": round(float(ic), 4), "p_val": round(float(p), 4), "n": int(len(g))}
    return out


def run_event_study(signal_path: Path = config.SIGNAL_PQ,
                    horizon: int = config.PRIMARY_HORIZON,
                    output_dir: Path = config.RESULTS_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel(signal_path)
    log.info(f"Loaded signal panel: {len(panel)} events, {panel['ticker'].nunique()} tickers")

    ic = compute_ic_series(panel, horizon)
    ic.to_csv(output_dir / "ic_series.csv", index=False)
    ls, bt = quintile_backtest(panel, horizon)
    if not ls.empty:
        ls.to_csv(output_dir / "quintile_returns.csv", index=False)
    res = {
        "horizon_days": horizon,
        "entry_lag_days": config.ENTRY_LAG,
        "ic_stats": compute_ic_stats(ic),
        "alpha_decay": compute_alpha_decay(panel),
        "event_time_ls": bt,
        "regime_ic": regime_ic(panel, horizon),
    }

    s = res["ic_stats"]
    log.info(f"IC {horizon}d: mean={s.get('mean_ic')}  ICIR={s.get('icir')}  "
             f"t(NW)={s.get('t_statistic_nw')}  IC>0={s.get('ic_positive_pct')}")
    for k, v in res["alpha_decay"].items():
        log.info(f"  decay {k:>4s}: IC={v['ic']:+.4f}  p={v['p_val']:.3f}")
    if bt:
        log.info(f"Event-time L/S: Sharpe={bt['sharpe_ratio']}  ann={bt['ann_return']:.2%}  "
                 f"t(NW)={bt['t_statistic_nw']}")
    for k, v in res["regime_ic"].items():
        log.info(f"  VIX {k:8s}: IC={v.get('ic')}  n={v.get('n')}")

    (output_dir / "event_study.json").write_text(json.dumps(res, indent=2, default=str))
    log.info(f"Saved to {output_dir / 'event_study.json'}")
    return res


def main():
    config.setup_logging()
    p = argparse.ArgumentParser(description="PEAD event study: IC, decay, L/S, regimes")
    p.add_argument("--signal-path", default=str(config.SIGNAL_PQ))
    p.add_argument("--horizon", type=int, default=config.PRIMARY_HORIZON)
    p.add_argument("--output-dir", default=str(config.RESULTS_DIR))
    a = p.parse_args()
    run_event_study(Path(a.signal_path), a.horizon, Path(a.output_dir))


if __name__ == "__main__":
    main()
