"""E8. Order flow imbalance: linear price impact in the simulated Markovian book.

Output: figures/fig7_ofi.png, results/e8_ofi.{json,tex}
"""
import numpy as np

from common import FIG, fmt, fmt_pm, write_numbers
from stochastic_lob import impact
from stochastic_lob import markov_lob as ml
from stochastic_lob.plotting import plt, save, use_style

N_EVENTS = 300_000
WINDOW = 10.0      # aggregation window (s)


def main():
    use_style()
    params = ml.power_law_params(L=30, lam1=1.9, a=0.5, theta0=0.5, mu=1.0)
    rec = ml.simulate_markov_lob(params, N_EVENTS, rng=81, burn_in=20_000)
    ok = (rec.q_bid > 0) & (rec.q_ask > 0)
    t, bp, bq, ap, aq = rec.t[ok], rec.best_bid[ok], rec.q_bid[ok], rec.best_ask[ok], rec.q_ask[ok]
    ofi, dmid = impact.aggregate_ofi(t, bp, bq, ap, aq, WINDOW)
    fit = impact.ols(ofi, dmid)

    # Stylized benchmark: if every level held D orders, a net flow e moves one
    # quote by e/D ticks and the mid by e/(2D) ticks.
    depth = 0.5 * (bq.mean() + aq.mean())
    stylized = 1.0 / (2.0 * depth)

    use = np.abs(ofi) <= np.percentile(np.abs(ofi), 99.5)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(ofi[use], dmid[use], ".", ms=1.5, alpha=0.35, color="0.45", rasterized=True,
            label=f"{WINDOW:.0f} s Windows")
    xx = np.linspace(ofi[use].min(), ofi[use].max(), 50)
    ax.plot(xx, fit["intercept"] + fit["slope"] * xx, "-", label="OLS Fit")
    ax.plot(xx, stylized * xx, "--", label=r"Flat Book, $1/(2\bar{q})$")
    ax.set_xlabel(r"Order Flow Imbalance $\mathrm{OFI}$ (orders)")
    ax.set_ylabel(r"Mid-Price Change $\Delta S/\delta$ (ticks)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.55 * (hi - lo))
    ax.legend(loc="upper left", markerscale=4)
    fig.tight_layout()
    save(fig, "fig7_ofi", FIG)

    write_numbers("e8_ofi", {
        "EeightWindow": fmt(WINDOW, 0),
        "EeightN": f"{fit['n']:,}".replace(",", "{,}"),
        "EeightSlope": fmt_pm(fit["slope"], fit["slope_se"], 4),
        "EeightRsq": fmt(fit["r2"], 2),
        "EeightDepth": fmt(depth, 2),
        "EeightStylized": fmt(stylized, 4),
        "EeightRelDiff": fmt(100 * abs(fit["slope"] / stylized - 1), 1),
    })


if __name__ == "__main__":
    main()
