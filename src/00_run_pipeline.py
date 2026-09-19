"""
00_run_pipeline.py
==================
Master runner for the PEAD (Post-Earnings Announcement Drift) Alpha Pipeline.

Usage:
    python src/00_run_pipeline.py                    # default 200 tickers, 21d horizon
    python src/00_run_pipeline.py --tickers 50       # quick test
    python src/00_run_pipeline.py --horizon 63       # 63-day horizon backtest
    python src/00_run_pipeline.py --tickers 200 --start-date 2010-01-01

Steps:
    1. Fetch earnings announcement dates from SEC EDGAR 8-K filings (PIT)
    2. Compute time-series SUE via AR(1) / Seasonal Random Walk (EDGAR XBRL EPS)
    3. Build signal panel: merge SUE + price data, compute multi-horizon excess returns
    4. IC analysis, alpha decay, L/S backtest, VIX regime conditioning

Author: Tingxuan Wu
"""

import sys
import json
import argparse
import logging
import importlib.util
import pandas as pd
from pathlib import Path

# ── Add src to path ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Lazy module loader ─────────────────────────────────────────────────────────
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_src = Path(__file__).parent

# ── Ticker universe ────────────────────────────────────────────────────────────
# 200 liquid US equities across 11 GICS sectors.
# Selection criteria: S&P 500 / Russell 1000, sufficient 8-K filing history
# (listed before 2012), active as of 2024. Covers multiple sectors to avoid
# sector-concentration bias in IC analysis.

_UNIVERSE_200 = [
    # ── Technology (40) ──────────────────────────────────────────────────────
    "AAPL","MSFT","NVDA","GOOGL","META","AVGO","ORCL","CSCO","ADBE","CRM",
    "AMD","INTC","QCOM","TXN","AMAT","KLAC","LRCX","MU","MRVL","ANET",
    "SNPS","CDNS","ANSS","FTNT","PANW","CRWD","ZBRA","KEYS","SWKS","QRVO",
    "CRUS","APH","TEL","GLW","JNPR","HPQ","HPE","WDC","STX","NTAP",

    # ── Communication Services (10) ───────────────────────────────────────────
    "NFLX","DIS","CMCSA","T","VZ","CHTR","TMUS","FOX","WBD","PARA",

    # ── Consumer Discretionary (20) ───────────────────────────────────────────
    "AMZN","TSLA","HD","MCD","NKE","SBUX","TJX","BKNG","LOW","GM",
    "F","ROST","ORLY","AZO","BBY","DG","DLTR","YUM","CMG","HLT",

    # ── Consumer Staples (15) ─────────────────────────────────────────────────
    "WMT","KO","PEP","COST","PG","PM","MO","CL","KMB","GIS",
    "CHD","LW","MKC","SJM","HRL",

    # ── Health Care (25) ─────────────────────────────────────────────────────
    "UNH","JNJ","LLY","ABT","MRK","ABBV","TMO","DHR","BMY","AMGN",
    "GILD","CVS","CI","HUM","IQV","WST","CTLT","IDXX","HOLX","TECH",
    "ISRG","SYK","BSX","MDT","ZBH",

    # ── Financials (20) ───────────────────────────────────────────────────────
    "JPM","BAC","WFC","GS","MS","BLK","SCHW","AXP","USB","PNC",
    "ICE","CME","CBOE","MKTX","LPLA","AFL","MET","PRU","ALL","TRV",

    # ── Industrials (25) ──────────────────────────────────────────────────────
    "HON","RTX","GE","LMT","NOC","GD","BA","EMR","ETN","PH",
    "ROK","ITW","CMI","PCAR","DE","CAT","MMM","DOV","XYL","ROP",
    "FTV","AME","VRSK","TDG","HEI",

    # ── Energy (15) ───────────────────────────────────────────────────────────
    "XOM","CVX","COP","EOG","SLB","HAL","BKR","MPC","VLO","PSX",
    "OXY","PXD","DVN","FANG","CLR",

    # ── Materials (10) ────────────────────────────────────────────────────────
    "LIN","APD","ECL","SHW","PPG","NEM","FCX","NUE","ALB","CF",

    # ── Real Estate (5) ───────────────────────────────────────────────────────
    "AMT","PLD","EQIX","CCI","PSA",

    # ── Utilities (5) ─────────────────────────────────────────────────────────
    "NEE","DUK","SO","AEP","EXC",

    # ── Extra mid-caps (10, for small-cap PEAD alpha) ─────────────────────────
    "LRCX","CLB","JBL","FLEX","ON","BKR","LHX","IQV","CTLT","WST",
]

# Deduplicate while preserving order
_seen = set()
UNIVERSE_200 = []
for t in _UNIVERSE_200:
    if t not in _seen:
        _seen.add(t)
        UNIVERSE_200.append(t)


# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).parent.parent
DATA_DIR    = _ROOT / "data"
RESULTS_DIR = _ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="PEAD Alpha Pipeline — Full Run")
    parser.add_argument("--tickers",    type=int, default=200,
                        help="Number of tickers from universe (default: 200)")
    parser.add_argument("--horizon",    type=int, default=21,
                        help="Primary return horizon in trading days (default: 21)")
    parser.add_argument("--start-date", default="2005-01-01",
                        help="Start date for price / announcement history")
    parser.add_argument("--end-date",   default="2026-09-30")
    parser.add_argument("--skip-step",  type=int, nargs="*", default=[],
                        help="Skip steps (e.g. --skip-step 1 2 to skip download + SUE)")
    args = parser.parse_args()

    tickers = UNIVERSE_200[:args.tickers]

    print("=" * 60)
    print("PEAD Alpha Pipeline — Post-Earnings Announcement Drift")
    print("=" * 60)
    print(f"  Ticker universe : {len(tickers)} equities")
    print(f"  Primary horizon : {args.horizon} trading days")
    print(f"  Date range      : {args.start_date} → {args.end_date}")
    print(f"  Skipping steps  : {args.skip_step or 'none'}")
    print()

    # ── Step 1: Earnings dates ────────────────────────────────────────────────
    earnings_csv = DATA_DIR / "edgar" / "earnings_dates.csv"

    if 1 not in args.skip_step:
        print("[Step 1/4] Fetching earnings announcement dates from SEC EDGAR...")
        m1 = _load("fetch_earnings_dates", _src / "01_fetch_earnings_dates.py")
        ann_df = m1.build_earnings_date_table(
            tickers     = tickers,
            start_date  = args.start_date,
            end_date    = args.end_date,
            output_path = earnings_csv,
        )
        print(f"  → {len(ann_df)} announcement events saved to {earnings_csv.name}")
    else:
        print("[Step 1/4] SKIPPED — using cached earnings_dates.csv")
        if not earnings_csv.exists():
            print("  [ERROR] Cache not found. Remove --skip-step 1 and rerun.")
            sys.exit(1)
        ann_df = pd.read_csv(earnings_csv)
        print(f"  → Loaded {len(ann_df)} events from cache")

    # ── Step 2: SUE computation ───────────────────────────────────────────────
    sue_csv = DATA_DIR / "signals" / "sue_panel.csv"

    if 2 not in args.skip_step:
        print("\n[Step 2/4] Computing time-series SUE (EDGAR XBRL EPS)...")
        m2 = _load("compute_sue", _src / "02_compute_sue.py")
        sue_df = m2.build_sue_panel(
            earnings_dates_path = earnings_csv,
            output_path         = sue_csv,
            use_ar1             = True,
        )
        n_sue = sue_df["sue"].notna().sum() if not sue_df.empty else 0
        pct   = n_sue / max(len(sue_df), 1)
        print(f"  → {n_sue}/{len(sue_df)} SUE values computed ({pct:.1%} coverage)")
    else:
        print("\n[Step 2/4] SKIPPED — using cached sue_panel.csv")

    # ── Step 3: Signal panel ──────────────────────────────────────────────────
    signal_pq  = DATA_DIR / "signals" / "signal_panel.parquet"
    signal_csv = DATA_DIR / "signals" / "signal_panel.csv"

    if 3 not in args.skip_step:
        print("\n[Step 3/4] Building signal panel (prices + excess returns)...")
        m3 = _load("build_signal", _src / "03_build_signal.py")
        panel = m3.build_signal_panel(
            sue_path   = sue_csv,
            output_pq  = signal_pq,
            output_csv = signal_csv,
            start_date = args.start_date,
            end_date   = args.end_date,
        )
        print(f"  → Signal panel: {len(panel)} events, {panel['ticker'].nunique()} tickers")
        if not panel.empty and "excess_ret_21d" in panel.columns:
            cov = panel["excess_ret_21d"].notna().mean()
            print(f"  → 21d excess return coverage: {cov:.1%}")
    else:
        print("\n[Step 3/4] SKIPPED — using cached signal_panel.parquet")

    # ── Step 4: Backtest ──────────────────────────────────────────────────────
    if 4 not in args.skip_step:
        print(f"\n[Step 4/4] Running IC analysis + L/S backtest (horizon={args.horizon}d)...")
        m4 = _load("backtest", _src / "04_backtest.py")
        results = m4.run_full_backtest(
            signal_path = signal_pq,
            horizon     = args.horizon,
            output_dir  = RESULTS_DIR,
        )
    else:
        print("\n[Step 4/4] SKIPPED")
        results_path = RESULTS_DIR / "backtest_results.json"
        results = json.loads(results_path.read_text()) if results_path.exists() else {}

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    ic   = results.get("ic_stats", {})
    bt   = results.get("backtest", {})
    dec  = results.get("alpha_decay", {})
    reg  = results.get("regime_ic", {})

    def _f(v, pct=False):
        if v is None or v == "N/A": return "N/A"
        return f"{v:.2%}" if pct else f"{v}"

    print(f"  Ticker universe : {len(tickers)}")
    print(f"  Primary horizon : {args.horizon}d")
    print()
    print(f"  IC (mean)       : {_f(ic.get('mean_ic'))}")
    print(f"  ICIR            : {_f(ic.get('icir'))}")
    print(f"  IC t-stat (NW)  : {_f(ic.get('t_statistic'))}")
    print(f"  IC > 0          : {_f(ic.get('ic_positive_pct'), pct=True)}")
    print()
    print(f"  Alpha decay:")
    for hk in ["1d","5d","21d","63d"]:
        v = dec.get(hk, {})
        if v:
            print(f"    {hk:4s}  IC={v.get('ic',0):+.3f}  p={v.get('p_val',1):.3f}")
    print()
    print(f"  L/S Sharpe      : {_f(bt.get('sharpe_ratio'))}")
    print(f"  L/S Ann. Return : {_f(bt.get('ann_return'), pct=True)}")
    print(f"  Max Drawdown    : {_f(bt.get('max_drawdown'), pct=True)}")
    print(f"  Calmar ratio    : {_f(bt.get('calmar_ratio'))}")
    print(f"  L/S t-stat (NW) : {_f(bt.get('t_statistic'))}")
    print()
    print(f"  VIX regime IC:")
    for name in ["stress","neutral","risk_on"]:
        v = reg.get(name, {})
        ic_val = v.get("ic")
        if ic_val is not None:
            print(f"    {name:8s}  IC={ic_val:+.3f}  p={v.get('p_val',1):.3f}  n={v.get('n',0)}")
    print()
    print(f"  Full results → {RESULTS_DIR}/backtest_results.json")
    print("=" * 60)
    print()
    print("  CV bullet point targets (after 200-ticker run):")
    print("    IC ≥ 0.03–0.05, t-stat > 2.5")
    print("    L/S Sharpe ≥ 0.8")
    print("    VIX stress IC significantly higher than full-sample IC")


if __name__ == "__main__":
    main()
