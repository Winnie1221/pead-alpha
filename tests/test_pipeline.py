"""
Offline unit tests on synthetic data. They pin down the point-in-time and
accounting rules that matter most for a backtest's credibility.

Run: pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from pead.earnings_dates import extract_earnings_events
from pead.signal_panel import compute_forward_returns
from pead.sue import expected_eps, map_events_to_quarters, quarterly_eps_from_facts
from pead.tradable_backtest import (build_active_signal_matrix, portfolio_returns,
                                    quintile_weights)


# ── Step 1: earnings dates ────────────────────────────────────────────────────

def test_only_item_202_8ks_are_earnings_events():
    filings = pd.DataFrame({
        "form": ["8-K", "8-K", "8-K/A", "10-Q", "10-Q"],
        "filing_date": ["2020-04-28", "2020-05-15", "2020-04-30", "2020-05-05", "2020-11-10"],
        "report_date": ["2020-04-28", "2020-05-14", "2020-04-28", "2020-03-31", "2020-09-30"],
        "items": ["2.02,9.01", "5.02", "2.02", "", ""],
        "accession": list("abcde"),
    })
    ev = extract_earnings_events(filings, "XYZ", "0000000001", "2019-01-01", "2021-01-01")
    # 8-K 5.02 (officer change) and the 8-K/A are ignored; Q1 10-Q is covered
    # by the 2.02 8-K; Q3 10-Q has no 8-K, so it becomes a fallback event.
    assert list(ev["source"]) == ["8K_item202", "10Q_fallback"]
    assert ev["announce_date"].iloc[0] == pd.Timestamp("2020-04-28")
    assert ev["announce_date"].iloc[1] == pd.Timestamp("2020-11-10")


# ── Step 2: SUE ───────────────────────────────────────────────────────────────

def _fact(start, end, val, filed, form, fp):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form, "fp": fp}


def test_ytd_rows_excluded_and_q4_derived():
    rows = [
        _fact("2021-01-01", "2021-03-31", 1.0, "2021-05-01", "10-Q", "Q1"),
        _fact("2021-04-01", "2021-06-30", 1.1, "2021-08-01", "10-Q", "Q2"),
        _fact("2021-01-01", "2021-06-30", 2.1, "2021-08-01", "10-Q", "Q2"),   # 6M YTD
        _fact("2021-07-01", "2021-09-30", 1.2, "2021-11-01", "10-Q", "Q3"),
        _fact("2021-01-01", "2021-09-30", 3.3, "2021-11-01", "10-Q", "Q3"),   # 9M YTD
        _fact("2021-01-01", "2021-12-31", 4.8, "2022-02-15", "10-K", "FY"),   # annual
    ]
    facts = {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": rows}}}}}
    q = quarterly_eps_from_facts(facts)
    assert list(q["eps"].round(2)) == [1.0, 1.1, 1.2, 1.5]   # Q4 = 4.8 − 3.3
    assert list(q["derived_q4"]) == [False, False, False, True]


def _eps_history(n=16, start="2016-03-31"):
    ends = pd.date_range(start, periods=n, freq="QE")
    return pd.DataFrame({
        "period_start": ends - pd.Timedelta(days=90),
        "period_end": ends,
        "eps": np.arange(n, dtype=float) * 0.1 + np.tile([0.0, 0.2, 0.1, 0.3], n // 4 + 1)[:n],
        "filed_date": ends + pd.Timedelta(days=35),
        "derived_q4": False,
    })


def test_expected_eps_uses_only_data_filed_before_announcement():
    eps = _eps_history()
    target = eps["period_end"].iloc[-1]
    ann = target + pd.Timedelta(days=25)
    base = expected_eps(eps, target, ann, use_ar1=False)
    # Change the current quarter's EPS (filed after the announcement):
    eps2 = eps.copy()
    eps2.loc[eps2.index[-1], "eps"] = 99.0
    assert expected_eps(eps2, target, ann, use_ar1=False) == base
    # Seasonal RW expectation = EPS four quarters earlier
    assert base["eps_expected"] == pytest.approx(eps["eps"].iloc[-5])


def test_8k_maps_to_latest_prior_quarter():
    eps = _eps_history()
    ev = pd.DataFrame({"announce_date": [pd.Timestamp("2019-07-25")],
                       "fiscal_period_end": [pd.NaT], "source": ["8K_item202"]})
    m = map_events_to_quarters(ev, eps)
    assert m["period_end"].iloc[0] == pd.Timestamp("2019-06-30")


# ── Step 3: forward returns ───────────────────────────────────────────────────

def test_forward_returns_start_after_the_announcement_reaction():
    days = pd.bdate_range("2021-01-04", periods=40)
    px = pd.DataFrame({"AAA": 100.0, "SPY": 100.0}, index=days)
    px.iloc[11:, 0] = 110.0     # +10% jump on day0 + 1 (after-close release)
    px.iloc[13:, 0] = 121.0     # +10% drift two days later
    ev = pd.DataFrame({"ticker": ["AAA"], "announce_date": [days[10]]})
    r = compute_forward_returns(px, ev, horizons=[1, 5], entry_lag=1)
    assert r["ann_ret"].iloc[0] == pytest.approx(0.10)       # reaction, not traded
    assert r["ret_1d"].iloc[0] == pytest.approx(0.0)
    assert r["ret_5d"].iloc[0] == pytest.approx(0.10)       # drift only
    assert r["excess_ret_5d"].iloc[0] == pytest.approx(0.10)


# ── Step 5: tradable backtest ────────────────────────────────────────────────

def test_book_cannot_earn_the_announcement_jump():
    days = pd.bdate_range("2021-01-04", periods=60)
    names = [f"S{i}" for i in range(10)]
    px = pd.DataFrame(100.0, index=days, columns=names)
    ann = days[20]
    px.loc[days[21]:, "S9"] = 120.0          # jump at the first close after the 8-K
    panel = pd.DataFrame({"ticker": names, "announce_date": ann,
                          "sue_winsor": np.arange(10, dtype=float)})
    active = build_active_signal_matrix(panel, days, hold_days=5, entry_lag=1)
    assert active.loc[days[20]].isna().all()     # not active on the filing day
    assert active.loc[days[21]].notna().all()    # active from day0 + 1 close
    assert active.loc[days[26]].isna().all()     # expires after hold_days
    w = quintile_weights(active, min_names=10)
    acct = portfolio_returns(w, px)
    assert acct["gross_ret"].abs().max() == pytest.approx(0.0)


def test_turnover_accounting():
    days = pd.bdate_range("2021-01-04", periods=5)
    px = pd.DataFrame({"A": 100.0, "B": 100.0}, index=days)
    w = pd.DataFrame({"A": [0, .5, .5, 0, 0], "B": [0, -.5, -.5, 0, 0]},
                     index=days, dtype=float)
    acct = portfolio_returns(w, px, costs_bps=[10.0])
    # open 100% gross (turnover 0.5), hold (0), close (0.5)
    assert list(acct["turnover"].round(6)) == [0.0, 0.5, 0.0, 0.5, 0.0]
    assert acct["net_ret_10bps"].sum() == pytest.approx(-1.0 * 10e-4)
