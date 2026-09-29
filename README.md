# PEAD Alpha: Post-Earnings Announcement Drift on Free Data

An end-to-end, point-in-time research pipeline for the post-earnings announcement drift (PEAD) anomaly on 189 liquid US equities, 2009-2026. It builds everything from public sources (SEC EDGAR and Yahoo Finance). No I/B/E/S, Compustat or CRSP.

The pipeline goes from raw SEC filings to a tradable daily long/short book, with turnover and transaction costs:

```
SEC EDGAR 8-K Item 2.02 ──► announcement dates (PIT)
SEC XBRL companyfacts ────► quarterly EPS ──► SUE (AR(1) / seasonal random walk)
Yahoo Finance prices ─────► post-entry excess returns
                               │
                               ├──► event study: rank IC, alpha decay, VIX regimes
                               └──► daily dollar-neutral quintile book: turnover, costs, subperiods
```

## Results

Sample: 189 US stocks, 2009-2026. 13,464 earnings events from 8-K Item 2.02 filings (99.1%; 0.9% 10-Q/10-K fallback), 11,352 with valid SUE (91.5%). Entry at the close one trading day after filing; primary horizon 21 trading days.

### Signal quality (quarterly rank IC)

| Mean IC | ICIR | t-stat (Newey-West) | IC > 0 | Quarters |
|---|---|---|---|---|
| 0.0554 | 0.63 | 5.58 | 68.80% | 64 |

### Alpha decay (pooled rank IC)

| Horizon | IC | p-value | Events |
|---|---|---|---|
| 1d | +0.0324 | 0.001 | 10,197 |
| 5d | +0.0387 | 0.000 | 10,196 |
| 21d | +0.0476 | 0.000 | 10,189 |
| 63d | +0.0447 | 0.000 | 10,034 |

### Tradable daily book (dollar-neutral quintile L/S)

| Scenario | Sharpe | Ann. return | Ann. vol | Max DD | Turnover (x/yr) | Breakeven (bps) |
|---|---|---|---|---|---|---|
| Gross | 0.39 | 2.71% | 6.88% | -20.12% | 29.0 | 9.4 |
| Net of 5 bps | 0.18 | 1.26% | 6.89% | -24.96% | 29.0 | 4.4 |
| Net of 10 bps | -0.03 | -0.19% | 6.89% | -29.52% | 29.0 | -0.6 |

### Subperiods (tradable book)

| Subperiod | Gross Sharpe | Net Sharpe (10 bps) |
|---|---|---|
| 2009-2015 | 0.82 | 0.30 |
| 2016-2019 | 0.69 | 0.12 |
| 2020-2026 | 0.12 | -0.25 |

Breakeven is the one-way cost that zeroes the gross return (gross row). The breakeven figures on the net rows are the *additional* cost the net-of-X book could bear.

### Event-time quintile spread (per quarterly cohort)

Mean Q5 - Q1 excess return over 21 days: 0.96% (t-stat NW 4.20, hit rate 65.60%, 64 quarters).

### IC by VIX regime

| Regime | IC | p-value | Events |
|---|---|---|---|
| stress | +0.0234 | 0.420 | 1188 |
| neutral | +0.0422 | 0.002 | 5428 |
| risk_on | +0.0686 | 0.000 | 3573 |

The figures below are generated in `results/figures/`.


## Method

### 1. Announcement dates: `pead/earnings_dates.py`

- The earnings date is the filing date of the 8-K tagged **Item 2.02** (*Results of Operations and Financial Condition*), mandatory since August 2004.
- Other 8-Ks (officer changes, M&A, financing) are not earnings events and are ignored.
- **Fallback:** if no Item 2.02 8-K was filed between a quarter's end and its 10-Q/10-K, that 10-Q/10-K filing date is used instead. These events are tagged `10Q_fallback`.
- Requests are rate-limited to 10 per second under the SEC fair-access policy, and every response is cached to disk.

### 2. Standardized unexpected earnings: `pead/sue.py`

$$\text{SUE}_t = \frac{EPS_t - E[EPS_t]}{\sigma_t}$$

- **Quarterly EPS from XBRL.** Only 3-month facts count as quarterly values. 10-Qs also report 6- and 9-month year-to-date EPS under the same fiscal-period tag, and those are filtered out by duration. Q4 is derived as FY − (Q1+Q2+Q3), because 10-Ks report annual EPS only. For restated values, the first filed value is kept.
- **Expectation models:**
  - AR(1) on seasonal differences, $\Delta EPS_t = \alpha + \beta\,\Delta EPS_{t-1}$, when at least 8 consecutive-quarter pairs exist.
  - Otherwise, a seasonal random walk (Bernard & Thomas, 1990): $E[EPS_t] = EPS_{t-4}$, with $\sigma$ the std of the last 8 seasonal differences.
- **Point-in-time.** Expectations and $\sigma$ use only quarters filed before the announcement date. Each 8-K is mapped to the latest fiscal quarter that ended before it.

### 3. Signal panel: `pead/signal_panel.py`

- **Timing.** Most earnings 8-Ks are filed after the close. The first tradable close is therefore **day 0 + 1**, and every forward return starts there.
- **Announcement reaction.** The move from close(day −1) to close(day 0 + 1) is stored separately as `ann_ret`. It is *not* counted as drift.
- **Excess returns.** Market-excess returns versus SPY at 1, 5, 21 and 63 trading days.

### 4. Event study: `pead/event_study.py`

- **Rank IC.** Quarterly Spearman IC of SUE against 21-day excess return, with ICIR, a Newey-West t-stat (4 lags) and the share of quarters with IC > 0.
- **Alpha decay.** Pooled IC across the four horizons.
- **Quintile spread.** Q5 − Q1 spread per quarterly cohort (event time).
- **VIX regimes.** IC split by VIX regime: stress ≥ 25, neutral 15–25, risk-on < 15.

### 5. Tradable backtest: `pead/tradable_backtest.py`

- **Holding window.** A name is active for 21 trading days, starting at its entry close.
- **Portfolio.** Every day, the active names are ranked by SUE: long the top quintile and short the bottom quintile. Positions are equal-weighted, 50% gross per leg, dollar-neutral, and rebalanced at the close. The book holds cash when fewer than 10 names are active.
- **Turnover.** One-way turnover is drift-adjusted: $\tfrac12\sum_i |w_{i,t} - w_{i,t-1}(1+r_{i,t})/(1+r_{p,t})|$.
- **Costs.** Net returns are reported at 5 and 10 bps one-way, along with the breakeven cost $E[r]/E[\text{turnover}]$.
- **Subperiods.** Results are also shown for 2009–2015, 2016–2019 and 2020–2026.

## Point-in-time safeguards

The unit tests in `tests/` pin these safeguards down:

| Risk | Safeguard | Test |
|---|---|---|
| Non-earnings 8-Ks treated as earnings dates, attaching EPS before it was public | Only Item 2.02 8-Ks are events | `test_only_item_202_8ks_are_earnings_events` |
| YTD (6M/9M) or annual EPS mistaken for a quarter | 3-month duration filter; Q4 = FY − 9M | `test_ytd_rows_excluded_and_q4_derived` |
| Expectation model sees the quarter being forecast | History restricted to filings before the announcement | `test_expected_eps_uses_only_data_filed_before_announcement` |
| Entering at the close before an after-hours release, earning the announcement jump | Entry at day 0 + 1 close in both the event study and the daily book | `test_forward_returns_start_after_the_announcement_reaction`, `test_book_cannot_earn_the_announcement_jump` |
| Wrong turnover or cost accounting | Drift-adjusted one-way turnover | `test_turnover_accounting` |

## Quick start

```bash
git clone https://github.com/Winnie1221/pead-alpha.git
cd pead-alpha
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# The SEC requires a User-Agent with contact details
export SEC_USER_AGENT="Your Name your.email@example.com"

python -m pead.pipeline --tickers 30     # smoke run, a few minutes
python -m pead.pipeline                  # full universe
python -m pead.pipeline --skip 1 2       # rerun analysis on cached EDGAR data

pytest -q                                # offline unit tests
```

Each step can also be run on its own: `python -m pead.earnings_dates`, `pead.sue`, `pead.signal_panel`, `pead.event_study`, `pead.tradable_backtest`, `pead.plots` and `pead.report`.

## Repository layout

```
pead/
  config.py             paths, timing conventions, constants
  universe.py           ~200-name ticker universe
  earnings_dates.py     step 1  SEC EDGAR 8-K Item 2.02 dates
  sue.py                step 2  XBRL EPS → SUE
  signal_panel.py       step 3  prices, forward and announcement returns
  event_study.py        step 4  IC, decay, quintile spread, VIX regimes
  tradable_backtest.py  step 5  daily book, turnover, costs
  plots.py, report.py   figures and results/SUMMARY.md
  pipeline.py           runs steps 1–6
tests/                  offline unit tests (synthetic data)
data/                   downloaded data and caches (git-ignored)
results/                JSON / CSV outputs and figures
```

## Limitations

- **Survivorship bias.** The universe is today's large caps. Names that were delisted or acquired during the sample are missing, which likely flatters results.
- **Large-cap universe.** PEAD is documented to be strongest in small, illiquid stocks. A large-cap universe is a conservative test of the anomaly, but it is also where costs are realistic.
- **Time-series SUE.** SUE is time-series based. Analyst-consensus SUE (I/B/E/S) is sharper but is not free.
- **XBRL coverage.** XBRL EPS begins around 2009–2011, so SUE coverage in earlier years is thin.
- **Filing-date timing.** The 8-K filing date is used as the event date. Releases before the open would allow an earlier entry, so the one-day entry lag is conservative for those names.
- **Q4 approximation.** Deriving Q4 as FY − 9M is exact only when the diluted share count is stable.
- **Prices.** Adjusted closes come from Yahoo Finance, with no borrow costs and no shorting constraints.

## References

- Ball, R., & Brown, P. (1968). An empirical evaluation of accounting income numbers. *Journal of Accounting Research*.
- Bernard, V., & Thomas, J. (1989, 1990). Post-earnings-announcement drift: delayed price response or risk premium?; Evidence that stock prices do not fully reflect the implications of current earnings for future earnings. *JAR*; *JAE*.
- Foster, G. (1977). Quarterly accounting data: time-series properties and predictive-ability results. *The Accounting Review*.
- Newey, W., & West, K. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*.

## License

MIT © Tingxuan Wu
