"""
Figures for the README, built from the files in results/.

  results/figures/cumulative_returns.png  — tradable book, gross vs net of costs
  results/figures/alpha_decay.png         — pooled IC by horizon
  results/figures/ic_series.png           — quarterly IC with 4-quarter mean
"""

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

log = logging.getLogger(__name__)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def plot_cumulative(results_dir=config.RESULTS_DIR, out_dir=config.FIGURES_DIR):
    df = pd.read_csv(results_dir / "tradable_daily_returns.csv", index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, label, color in [("gross_ret", "Gross", BLUE),
                              ("net_ret_5bps", "Net, 5 bps", ORANGE),
                              ("net_ret_10bps", "Net, 10 bps", AQUA)]:
        if col in df:
            cum = (1 + df[col]).cumprod() - 1
            ax.plot(cum.index, cum * 100, color=color, linewidth=2, label=label)
            ax.annotate(f"{label} {cum.iloc[-1]:+.0%}", (cum.index[-1], cum.iloc[-1] * 100),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK)
    _style(ax, "PEAD quintile long/short, cumulative return (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_returns.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_decay(results_dir=config.RESULTS_DIR, out_dir=config.FIGURES_DIR):
    res = json.loads((results_dir / "event_study.json").read_text())
    dec = res.get("alpha_decay", {})
    if not dec:
        return
    keys = list(dec)
    ics = [dec[k]["ic"] for k in keys]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(keys, ics, color=BLUE, width=0.55)
    for b, v in zip(bars, ics):
        ax.annotate(f"{v:+.3f}", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4 if v >= 0 else -12), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    _style(ax, "Pooled rank IC of SUE vs. excess return, by horizon")
    ax.set_xlabel("Holding horizon (trading days after entry)", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "alpha_decay.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_ic_series(results_dir=config.RESULTS_DIR, out_dir=config.FIGURES_DIR):
    ic = pd.read_csv(results_dir / "ic_series.csv")
    if ic.empty:
        return
    ic["date"] = pd.PeriodIndex(ic["period"], freq="Q").to_timestamp()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(ic["date"], ic["ic"], width=70, color=BLUE, alpha=0.55, label="Quarterly IC")
    ax.plot(ic["date"], ic["ic"].rolling(4).mean(), color=ORANGE, linewidth=2,
            label="4-quarter mean")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    _style(ax, "Quarterly rank IC, SUE vs. 21-day excess return")
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "ic_series.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_all_figures():
    config.ensure_dirs()
    for fn in (plot_cumulative, plot_alpha_decay, plot_ic_series):
        try:
            fn()
        except FileNotFoundError as e:
            log.warning(f"Skipped {fn.__name__}: {e}")
    log.info(f"Figures saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    config.setup_logging()
    make_all_figures()
