"""
Shared configuration: paths, SEC access settings and research constants.

Every module imports from here so that a timing convention (e.g. the entry
lag) can never silently differ between the event study and the tradable
backtest.
"""

import logging
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EDGAR_DIR = DATA_DIR / "edgar"
CACHE_DIR = EDGAR_DIR / "cache"
SIGNALS_DIR = DATA_DIR / "signals"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

EARNINGS_CSV = EDGAR_DIR / "earnings_dates.csv"
SUE_CSV = SIGNALS_DIR / "sue_panel.csv"
SIGNAL_PQ = SIGNALS_DIR / "signal_panel.parquet"


def ensure_dirs() -> None:
    for d in (EDGAR_DIR, CACHE_DIR, SIGNALS_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ── SEC EDGAR access ──────────────────────────────────────────────────────────
# The SEC fair-access policy requires a descriptive User-Agent with contact
# details. Set it through the environment instead of hard-coding an email:
#   export SEC_USER_AGENT="Your Name your.email@example.com"
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "")
SEC_MIN_INTERVAL = 0.11  # seconds between requests (≤ 10 req/s)


def sec_headers() -> dict:
    if not SEC_USER_AGENT:
        raise RuntimeError(
            "SEC_USER_AGENT is not set. The SEC requires contact details, e.g.\n"
            '  export SEC_USER_AGENT="Jane Doe jane.doe@example.com"'
        )
    return {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


# ── Research constants ────────────────────────────────────────────────────────
DEFAULT_START = "2009-01-01"
DEFAULT_END = "2026-09-30"

# Timing. Earnings 8-Ks are usually filed after the close, so the close of
# the filing day (day 0) is NOT tradable on the news. The first close at which
# we can act is day 0 + ENTRY_LAG. Forward returns and the daily backtest both
# start from that close.
ENTRY_LAG = 1

HORIZONS = [1, 5, 21, 63]  # forward-return horizons, trading days
PRIMARY_HORIZON = 21
HOLD_DAYS = 21  # holding window in the tradable backtest
MIN_NAMES = 10  # min active names before the tradable book takes positions
TRADING_DAYS = 252
COST_SCENARIOS_BPS = [5.0, 10.0]  # one-way transaction cost scenarios
NW_LAGS = 4  # Newey-West lags for the quarterly IC series

MARKET_TICKER = "SPY"
VIX_TICKER = "^VIX"

SUBPERIODS = {
    "2009-2015": ("2009-01-01", "2015-12-31"),
    "2016-2019": ("2016-01-01", "2019-12-31"),
    "2020-2026": ("2020-01-01", "2026-12-31"),
}


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
