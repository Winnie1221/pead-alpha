# PEAD Alpha & Regime Research

Point-in-time post-earnings announcement drift signal across 188 liquid 
S&P 500 / Russell 1000 equities (2007–2026, 41K+ earnings events).

## Key Results

| Metric | Value |
|--------|-------|
| Spearman IC | 0.042 |
| NW-HAC t-stat | 3.87 (p < 0.001) |
| ICIR | 0.57 |
| IC > 0 | 69% of cycles |
| L/S Sharpe | 0.98 |
| Max Drawdown | −8.72% |
| Calmar Ratio | 0.81 |

**Regime conditioning:** risk-on IC = +0.057 (p < 0.01) vs. stress IC = +0.015 (n.s.)

## Methodology

- SUE estimation via seasonal AR(1) with `filed_date` cutoff (PIT-compliant)
- 3-tier PIT source hierarchy: 8-K Item 2.02 > 8-K no-item > 10-Q fallback
- Newey-West HAC standard errors (lags = 4)
- Quintile L/S backtest, alpha decay 1d → 63d, VIX regime conditioning

## Usage

```bash
pip install -r requirements.txt
python src/00_run_pipeline.py --tickers 200 --horizon 21
```
