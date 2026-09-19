"""
04_backtest.py
==============
Module 4 of the PEAD Alpha Pipeline.

Purpose:
    Rigorous statistical evaluation of the PEAD signal (SUE → excess returns).

Analyses:
    1. IC Time Series
       - Monthly/quarterly Spearman rank IC: corr(SUE_t, excess_ret_Nd)
       - ICIR = mean(IC) / std(IC)
       - Newey-West t-statistic (lags=4) for autocorrelation-robust inference
       - % of periods with IC > 0

    2. Alpha Decay Analysis
       - IC at horizons 1d, 5d, 21d, 63d
       - Shows how quickly the PEAD signal decays (CV talking point)

    3. Quintile L/S Backtest (21-day holding period)
       - Sort events into SUE quintiles each period
       - Long Q5 (highest SUE), Short Q1 (lowest SUE)
       - Compute monthly L/S excess returns
       - Metrics: Sharpe ratio, annualized return, max drawdown, calmar ratio

    4. Regime Conditioning (VIX-based)
       - Split into stress (VIX>25) / neutral / risk-on (VIX<15)
       - Compare IC across regimes (PEAD stronger in low-vol environments)

    5. Results JSON for CV bullet points

Output:
    results/backtest_results.json
    results/ic_series.csv
    results/quintile_returns.csv

Author: Tingxuan Wu
"""

import json
import warnings
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).parent.parent
DATA_DIR    = _ROOT / "data"
SIGNALS_DIR = DATA_DIR / "signals"
RESULTS_DIR = _ROOT / "results"
SIGNAL_PQ   = SIGNALS_DIR / "signal_panel.parquet"
SIGNAL_CSV  = SIGNALS_DIR / "signal_panel.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
HORIZONS      = [1, 5, 21, 63]
PRIMARY_H     = 21          # primary horizon for L/S backtest
TRADING_DAYS  = 252
NW_LAGS       = 4           # Newey-West lags for quarterly IC series


# ── Helper: Newey-West standard error ─────────────────────────────────────────

def newey_west_se(x: np.ndarray, lags: int = NW_LAGS) -> float:
    """
    Newey-West HAC standard error for a 1-D time series x.
    Accounts for autocorrelation up to `lags` periods.
    """
    n   = len(x)
    xd  = x - x.mean()
    var = np.dot(xd, xd) / n        # variance

    for l in range(1, lags + 1):
        w    = 1 - l / (lags + 1)   # Bartlett kernel weight
        cov  = np.dot(xd[l:], xd[:-l]) / n
        var += 2 * w * cov

    return np.sqrt(max(var, 1e-12) / n)


# ── Analysis 1: IC Time Series ─────────────────────────────────────────────────

def compute_ic_series(
    panel: pd.DataFrame,
    horizon: int = PRIMARY_H,
    freq: str = "Q",
) -> pd.DataFrame:
    """
    Compute period-by-period Spearman IC: corr(sue_winsor, excess_ret_Nd).

    Assigns each event to the calendar period of its announce_date.
    Returns DataFrame: period, ic, n_obs.
    """
    ret_col = f"excess_ret_{horizon}d"
    df = panel[["announce_date", "sue_winsor", ret_col]].dropna().copy()
    df["period"] = df["announce_date"].dt.to_period(freq)

    records = []
    for period, grp in df.groupby("period"):
        if len(grp) < 3:
            continue
        ic, _ = stats.spearmanr(grp["sue_winsor"], grp[ret_col])
        records.append({"period": str(period), "ic": ic, "n_obs": len(grp)})

    return pd.DataFrame(records)


def compute_ic_stats(ic_series: pd.DataFrame) -> dict:
    """
    Aggregate IC time series into summary statistics with Newey-West inference.
    """
    ics = ic_series["ic"].dropna().values
    if len(ics) == 0:
        return {}

    mean_ic = ics.mean()
    std_ic  = ics.std(ddof=1)
    icir    = mean_ic / std_ic if std_ic > 0 else np.nan
    nw_se   = newey_west_se(ics)
    t_stat  = mean_ic / nw_se if nw_se > 0 else np.nan
    p_val   = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(ics) - 1)) if not np.isnan(t_stat) else np.nan
    ic_pos  = (ics > 0).mean()

    return {
        "mean_ic":        round(float(mean_ic), 4),
        "std_ic":         round(float(std_ic), 4),
        "icir":           round(float(icir), 3),
        "t_statistic":    round(float(t_stat), 3),
        "p_value":        round(float(p_val), 4),
        "ic_positive_pct":round(float(ic_pos), 3),
        "n_periods":      int(len(ics)),
    }


# ── Analysis 2: Alpha Decay ────────────────────────────────────────────────────

def compute_alpha_decay(panel: pd.DataFrame) -> dict:
    """
    IC and t-stat at each horizon. Shows rate of alpha decay.
    """
    decay = {}
    for h in HORIZONS:
        ret_col = f"excess_ret_{h}d"
        df = panel[["sue_winsor", ret_col]].dropna()
        if len(df) < 10:
            continue
        ic, pval = stats.spearmanr(df["sue_winsor"], df[ret_col])
        decay[f"{h}d"] = {
            "ic":    round(float(ic), 4),
            "p_val": round(float(pval), 4),
            "n":     int(len(df)),
        }
    return decay


# ── Analysis 3: Quintile L/S Backtest ─────────────────────────────────────────

def quintile_backtest(
    panel: pd.DataFrame,
    horizon: int = PRIMARY_H,
    freq: str = "Q",
) -> tuple:
    """
    Sort events into SUE quintiles each period.
    Long Q5 (top SUE), Short Q1 (bottom SUE).
    Returns (ls_returns_series, stats_dict).
    """
    ret_col = f"excess_ret_{horizon}d"
    df = panel[["announce_date", "ticker", "sue_winsor", ret_col]].dropna().copy()
    df["period"] = df["announce_date"].dt.to_period(freq)

    ls_returns = []
    for period, grp in df.groupby("period"):
        if len(grp) < 5:          # need at least 1 per quintile
            continue
        grp = grp.copy()
        grp["rank"] = grp["sue_winsor"].rank(pct=True)
        grp["quintile"] = pd.cut(grp["rank"],
                                  bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                  labels=[1, 2, 3, 4, 5], include_lowest=True)
        q5 = grp[grp["quintile"] == 5][ret_col].mean()
        q1 = grp[grp["quintile"] == 1][ret_col].mean()
        ls = q5 - q1
        ls_returns.append({
            "period":    str(period),
            "ls_return": ls,
            "q5_return": q5,
            "q1_return": q1,
            "n_events":  len(grp),
        })

    if not ls_returns:
        return pd.DataFrame(), {}

    ls_df = pd.DataFrame(ls_returns)
    ls_arr = ls_df["ls_return"].dropna().values

    # Annualize: each period is ~1 quarter (63 trading days)
    # L/S holding period = `horizon` trading days
    # Number of turns per year ≈ 252 / horizon
    turns_per_year = TRADING_DAYS / horizon
    ann_return = ls_arr.mean() * turns_per_year
    ann_vol    = ls_arr.std(ddof=1) * np.sqrt(turns_per_year)
    sharpe     = ann_return / ann_vol if ann_vol > 0 else np.nan

    # Max drawdown on cumulative L/S
    cumret  = np.cumprod(1 + ls_arr)
    peak    = np.maximum.accumulate(cumret)
    dd      = (cumret - peak) / peak
    max_dd  = dd.min()
    calmar  = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    # t-stat on mean L/S return (Newey-West)
    nw_se  = newey_west_se(ls_arr)
    t_stat = ls_arr.mean() / nw_se if nw_se > 0 else np.nan

    bt_stats = {
        "ann_return":    round(float(ann_return), 4),
        "ann_vol":       round(float(ann_vol), 4),
        "sharpe_ratio":  round(float(sharpe), 3),
        "max_drawdown":  round(float(max_dd), 4),
        "calmar_ratio":  round(float(calmar), 3),
        "t_statistic":   round(float(t_stat), 3),
        "n_periods":     int(len(ls_arr)),
        "horizon_days":  horizon,
    }
    return ls_df, bt_stats


# ── Analysis 4: VIX Regime Conditioning ───────────────────────────────────────

def regime_ic(panel: pd.DataFrame, horizon: int = PRIMARY_H) -> dict:
    """
    Download VIX and split events into regimes:
        stress:   VIX >= 25
        neutral:  15 <= VIX < 25
        risk_on:  VIX < 15

    Compute IC within each regime.
    """
    try:
        import yfinance as yf
        vix = yf.download("^VIX", start="2005-01-01", end="2026-09-30",
                          auto_adjust=True, progress=False)["Close"]
        vix.index = pd.to_datetime(vix.index)
        vix = vix.squeeze()
    except Exception as e:
        log.warning(f"VIX download failed: {e}")
        return {}

    ret_col = f"excess_ret_{horizon}d"
    df = panel[["announce_date", "sue_winsor", ret_col]].dropna().copy()
    df["vix"] = df["announce_date"].map(
        lambda d: vix.asof(d) if d in vix.index or d > vix.index[0] else np.nan
    )
    df = df.dropna(subset=["vix"])

    regimes = {
        "stress":   df[df["vix"] >= 25],
        "neutral":  df[(df["vix"] >= 15) & (df["vix"] < 25)],
        "risk_on":  df[df["vix"] < 15],
    }

    result = {}
    for name, grp in regimes.items():
        if len(grp) < 5:
            result[name] = {"ic": None, "n": len(grp)}
            continue
        ic, pval = stats.spearmanr(grp["sue_winsor"], grp[ret_col])
        result[name] = {
            "ic":    round(float(ic), 4),
            "p_val": round(float(pval), 4),
            "n":     int(len(grp)),
        }
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def run_full_backtest(
    signal_path: Path = SIGNAL_PQ,
    horizon: int      = PRIMARY_H,
    output_dir: Path  = RESULTS_DIR,
) -> dict:
    """
    Run all analyses and save results.
    """
    # Load signal panel
    if signal_path.suffix == ".parquet" and signal_path.exists():
        panel = pd.read_parquet(signal_path)
    else:
        csv_path = signal_path.with_suffix(".csv")
        if not csv_path.exists():
            csv_path = SIGNAL_CSV
        panel = pd.read_csv(csv_path, parse_dates=["announce_date"])

    panel["announce_date"] = pd.to_datetime(panel["announce_date"])
    log.info(f"Loaded signal panel: {len(panel)} events, {panel['ticker'].nunique()} tickers")

    results = {}

    # ── 1. IC Series ──────────────────────────────────────────────────────────
    log.info(f"\n[1/4] IC analysis (horizon={horizon}d)...")
    ic_df = compute_ic_series(panel, horizon=horizon)
    ic_stats = compute_ic_stats(ic_df)
    results["ic_stats"] = ic_stats
    ic_df.to_csv(output_dir / "ic_series.csv", index=False)

    log.info(f"  Mean IC:    {ic_stats.get('mean_ic', 'N/A')}")
    log.info(f"  ICIR:       {ic_stats.get('icir', 'N/A')}")
    log.info(f"  t-stat(NW): {ic_stats.get('t_statistic', 'N/A')}")
    log.info(f"  p-value:    {ic_stats.get('p_value', 'N/A')}")
    log.info(f"  IC>0:       {ic_stats.get('ic_positive_pct', 'N/A'):.0%}" if ic_stats.get('ic_positive_pct') is not None else "  IC>0: N/A")

    # ── 2. Alpha Decay ────────────────────────────────────────────────────────
    log.info(f"\n[2/4] Alpha decay analysis...")
    decay = compute_alpha_decay(panel)
    results["alpha_decay"] = decay
    for h_key, v in decay.items():
        log.info(f"  {h_key:4s}:  IC={v['ic']:+.3f}  p={v['p_val']:.3f}  n={v['n']}")

    # ── 3. Quintile L/S Backtest ──────────────────────────────────────────────
    log.info(f"\n[3/4] Quintile L/S backtest (horizon={horizon}d)...")
    ls_df, bt_stats = quintile_backtest(panel, horizon=horizon)
    results["backtest"] = bt_stats
    if not ls_df.empty:
        ls_df.to_csv(output_dir / "quintile_returns.csv", index=False)
        log.info(f"  Ann. Return: {bt_stats.get('ann_return', 0):.2%}")
        log.info(f"  Ann. Vol:    {bt_stats.get('ann_vol', 0):.2%}")
        log.info(f"  Sharpe:      {bt_stats.get('sharpe_ratio', 'N/A')}")
        log.info(f"  Max DD:      {bt_stats.get('max_drawdown', 0):.2%}")
        log.info(f"  Calmar:      {bt_stats.get('calmar_ratio', 'N/A')}")
        log.info(f"  t-stat(NW):  {bt_stats.get('t_statistic', 'N/A')}")

    # ── 4. Regime Conditioning ────────────────────────────────────────────────
    log.info(f"\n[4/4] Regime conditioning (VIX)...")
    regime = regime_ic(panel, horizon=horizon)
    results["regime_ic"] = regime
    for name, v in regime.items():
        if v.get("ic") is not None:
            log.info(f"  {name:8s}: IC={v['ic']:+.3f}  p={v['p_val']:.3f}  n={v['n']}")
        else:
            log.info(f"  {name:8s}: n={v.get('n', 0)} (insufficient)")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info("RESULTS SUMMARY (CV bullet points):")
    log.info(f"{'='*55}")
    ic_s = results.get("ic_stats", {})
    bt   = results.get("backtest", {})
    log.info(f"  IC (mean):          {ic_s.get('mean_ic', 'N/A')}")
    log.info(f"  ICIR:               {ic_s.get('icir', 'N/A')}")
    log.info(f"  IC t-stat (NW):     {ic_s.get('t_statistic', 'N/A')}")
    ic_pos = ic_s.get('ic_positive_pct')
    log.info(f"  IC > 0:             {f'{ic_pos:.0%}' if ic_pos else 'N/A'}")
    log.info(f"  L/S Sharpe:         {bt.get('sharpe_ratio', 'N/A')}")
    ann_r = bt.get('ann_return')
    mdd   = bt.get('max_drawdown')
    log.info(f"  L/S Ann. Return:    {f'{ann_r:.2%}' if ann_r is not None else 'N/A'}")
    log.info(f"  Max Drawdown:       {f'{mdd:.2%}' if mdd is not None else 'N/A'}")
    log.info(f"{'='*55}")

    # Save JSON
    out_path = output_dir / "backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Full results saved to {out_path}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PEAD factor backtest and IC analysis")
    parser.add_argument("--signal-path", default=str(SIGNAL_PQ))
    parser.add_argument("--horizon",     type=int, default=PRIMARY_H,
                        help="Primary return horizon in trading days (default: 21)")
    parser.add_argument("--output-dir",  default=str(RESULTS_DIR))
    args = parser.parse_args()

    results = run_full_backtest(
        signal_path = Path(args.signal_path),
        horizon     = args.horizon,
        output_dir  = Path(args.output_dir),
    )
