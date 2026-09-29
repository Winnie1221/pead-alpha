# Results summary

Entry lag: 1 trading day(s) after the 8-K filing date. Primary horizon: 21 trading days.

## Signal quality (quarterly rank IC)

| Mean IC | ICIR | t-stat (Newey-West) | IC > 0 | Quarters |
|---|---|---|---|---|
| 0.0554 | 0.63 | 5.58 | 68.80% | 64 |

## Alpha decay (pooled rank IC)

| Horizon | IC | p-value | Events |
|---|---|---|---|
| 1d | +0.0324 | 0.001 | 10,197 |
| 5d | +0.0387 | 0.000 | 10,196 |
| 21d | +0.0476 | 0.000 | 10,189 |
| 63d | +0.0447 | 0.000 | 10,034 |

## Tradable daily book (dollar-neutral quintile L/S)

| Scenario | Sharpe | Ann. return | Ann. vol | Max DD | Turnover (x/yr) | Breakeven (bps) |
|---|---|---|---|---|---|---|
| Gross | 0.39 | 2.71% | 6.88% | -20.12% | 29.0 | 9.4 |
| Net of 5 bps | 0.18 | 1.26% | 6.89% | -24.96% | 29.0 | 4.4 |
| Net of 10 bps | -0.03 | -0.19% | 6.89% | -29.52% | 29.0 | -0.6 |

| Subperiod | Gross Sharpe | Net Sharpe (10 bps) |
|---|---|---|
| 2009-2015 | 0.82 | 0.30 |
| 2016-2019 | 0.69 | 0.12 |
| 2020-2026 | 0.12 | -0.25 |

## Event-time quintile spread (per quarterly cohort)

Mean Q5 − Q1 excess return over 21 days: 0.96% (t-stat NW 4.20, hit rate 65.60%, 64 quarters).

## IC by VIX regime

| Regime | IC | p-value | Events |
|---|---|---|---|
| stress | +0.0234 | 0.420 | 1188 |
| neutral | +0.0422 | 0.002 | 5428 |
| risk_on | +0.0686 | 0.000 | 3573 |
