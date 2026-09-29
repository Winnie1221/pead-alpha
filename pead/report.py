"""Write results/SUMMARY.md (markdown tables) from the JSON outputs."""

import json
import logging

from . import config

log = logging.getLogger(__name__)


def _pct(v):
    return "–" if v is None else f"{v:.2%}"


def _num(v, fmt="{:.2f}"):
    return "–" if v is None else fmt.format(v)


def write_summary(results_dir=config.RESULTS_DIR) -> str:
    es = json.loads((results_dir / "event_study.json").read_text())
    tb = json.loads((results_dir / "tradable_backtest.json").read_text())
    ic, ls = es.get("ic_stats", {}), es.get("event_time_ls", {})
    lines = [
        "# Results summary", "",
        f"Entry lag: {es.get('entry_lag_days')} trading day(s) after the 8-K filing date. "
        f"Primary horizon: {es.get('horizon_days')} trading days.", "",
        "## Signal quality (quarterly rank IC)", "",
        "| Mean IC | ICIR | t-stat (Newey-West) | IC > 0 | Quarters |",
        "|---|---|---|---|---|",
        f"| {_num(ic.get('mean_ic'), '{:.4f}')} | {_num(ic.get('icir'))} | "
        f"{_num(ic.get('t_statistic_nw'))} | {_pct(ic.get('ic_positive_pct'))} | "
        f"{ic.get('n_periods', '–')} |", "",
        "## Alpha decay (pooled rank IC)", "",
        "| Horizon | IC | p-value | Events |", "|---|---|---|---|",
    ]
    for k, v in es.get("alpha_decay", {}).items():
        lines.append(f"| {k} | {v['ic']:+.4f} | {v['p_val']:.3f} | {v['n']:,} |")
    lines += ["", "## Tradable daily book (dollar-neutral quintile L/S)", "",
              "| Scenario | Sharpe | Ann. return | Ann. vol | Max DD | Turnover (x/yr) | Breakeven (bps) |",
              "|---|---|---|---|---|---|---|"]
    for k, label in [("gross", "Gross"), ("net_ret_5bps", "Net of 5 bps"),
                     ("net_ret_10bps", "Net of 10 bps")]:
        s = tb.get(k, {})
        lines.append(f"| {label} | {_num(s.get('sharpe'))} | {_pct(s.get('ann_return'))} | "
                     f"{_pct(s.get('ann_vol'))} | {_pct(s.get('max_drawdown'))} | "
                     f"{_num(s.get('ann_turnover_x'), '{:.1f}')} | "
                     f"{_num(s.get('breakeven_cost_bps'), '{:.1f}')} |")
    lines += ["", "| Subperiod | Gross Sharpe | Net Sharpe (10 bps) |", "|---|---|---|"]
    for k, v in tb.get("subperiods", {}).items():
        lines.append(f"| {k} | {_num(v['gross'].get('sharpe'))} | "
                     f"{_num(v['net_10bps'].get('sharpe'))} |")
    if ls:
        lines += ["", "## Event-time quintile spread (per quarterly cohort)", "",
                  f"Mean Q5 − Q1 excess return over {ls.get('horizon_days')} days: "
                  f"{_pct(ls.get('mean_period_ls_return'))} "
                  f"(t-stat NW {_num(ls.get('t_statistic_nw'))}, "
                  f"hit rate {_pct(ls.get('hit_rate'))}, {ls.get('n_periods')} quarters)."]
    reg = es.get("regime_ic", {})
    if reg:
        lines += ["", "## IC by VIX regime", "", "| Regime | IC | p-value | Events |",
                  "|---|---|---|---|"]
        for k, v in reg.items():
            lines.append(f"| {k} | {_num(v.get('ic'), '{:+.4f}')} | "
                         f"{_num(v.get('p_val'), '{:.3f}')} | {v.get('n', '–')} |")
    text = "\n".join(lines) + "\n"
    (results_dir / "SUMMARY.md").write_text(text)
    log.info(f"Wrote {results_dir / 'SUMMARY.md'}")
    return text


if __name__ == "__main__":
    config.setup_logging()
    print(write_summary())
