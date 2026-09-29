"""
Run the full PEAD pipeline end to end.

    python -m pead.pipeline                       # full 200-name universe
    python -m pead.pipeline --tickers 30          # quick smoke run
    python -m pead.pipeline --skip 1 2            # reuse cached EDGAR data

Steps
    1  earnings_dates     8-K Item 2.02 announcement dates (SEC EDGAR)
    2  sue                time-series SUE from XBRL EPS (AR(1) / seasonal RW)
    3  signal_panel       prices + forward excess returns
    4  event_study        IC, alpha decay, event-time L/S, VIX regimes
    5  tradable_backtest  daily dollar-neutral book with turnover and costs
    6  plots / report     README figures and results/SUMMARY.md
"""

import argparse
import logging

from . import config
from .earnings_dates import build_earnings_date_table
from .event_study import run_event_study
from .plots import make_all_figures
from .report import write_summary
from .signal_panel import build_signal_panel
from .sue import build_sue_panel
from .tradable_backtest import run_tradable_backtest
from .universe import UNIVERSE

log = logging.getLogger(__name__)


def main():
    config.setup_logging()
    p = argparse.ArgumentParser(description="PEAD alpha pipeline")
    p.add_argument("--tickers", type=int, default=len(UNIVERSE),
                   help="use the first N names of the universe")
    p.add_argument("--horizon", type=int, default=config.PRIMARY_HORIZON)
    p.add_argument("--start-date", default=config.DEFAULT_START)
    p.add_argument("--end-date", default=config.DEFAULT_END)
    p.add_argument("--skip", type=int, nargs="*", default=[],
                   help="step numbers to skip, reusing cached outputs")
    a = p.parse_args()
    config.ensure_dirs()
    tickers = UNIVERSE[:a.tickers]

    steps = [
        (1, "Earnings dates (SEC EDGAR 8-K Item 2.02)",
         lambda: build_earnings_date_table(tickers, a.start_date, a.end_date)),
        (2, "SUE (XBRL EPS)", lambda: build_sue_panel()),
        (3, "Signal panel (prices + forward returns)",
         lambda: build_signal_panel(start_date=a.start_date, end_date=a.end_date)),
        (4, "Event study", lambda: run_event_study(horizon=a.horizon)),
        (5, "Tradable backtest", lambda: run_tradable_backtest(end_date=a.end_date)),
        (6, "Figures and summary", lambda: (make_all_figures(), write_summary())),
    ]
    for n, name, fn in steps:
        if n in a.skip:
            log.info(f"[{n}/6] {name}: skipped")
            continue
        log.info(f"[{n}/6] {name}")
        fn()
    log.info(f"Done. Results in {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
