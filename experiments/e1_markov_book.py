"""E1. Markovian book (CST) and zero-intelligence book: depth profile and spread.

Output: figures/fig1_markov_book.png, results/e1_markov_book.{json,tex}
"""
import numpy as np

from common import FIG, fmt, fmt_pm, write_numbers
from stochastic_lob import markov_lob as ml
from stochastic_lob.plotting import WIDE, panel_label, plt, save, use_style

N_EVENTS = 300_000
BURN_IN = 20_000


def main():
    use_style()
    cst = ml.power_law_params(L=30, lam1=1.9, a=0.5, theta0=0.5, mu=1.0)
    zi = ml.zero_intelligence_params(L=30, lam0=0.6, theta0=0.2, mu=2.0)
    r_cst = ml.simulate_markov_lob(cst, N_EVENTS, rng=11, burn_in=BURN_IN)
    r_zi = ml.simulate_markov_lob(zi, N_EVENTS, rng=12, burn_in=BURN_IN)

    i = np.arange(1, cst.L + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    a1.plot(i, r_cst.depth_profile, "o-", ms=3, label="Power Law, Simulated")
    a1.plot(i, cst.far_depth(), "--", color="0.35", label=r"Power Law, $\lambda(i)/\theta(i)$")
    a1.plot(i, r_zi.depth_profile, "s-", ms=3, label="Zero Intelligence, Simulated")
    a1.plot(i, zi.far_depth(), ":", color="0.35", label=r"Zero Intelligence, $\lambda/\theta$")
    a1.set_xlabel(r"Distance to Opposite Best Quote $i$ (ticks)")
    a1.set_ylabel(r"Mean Depth $\bar{Q}(i)$ (orders)")
    a1.set_xlim(0, cst.L + 1)
    a1.set_ylim(0, 1.6 * max(r_cst.depth_profile.max(), r_zi.depth_profile.max(), zi.far_depth().max()))
    a1.legend(loc="upper right", ncol=2, fontsize=7.5)
    panel_label(a1, "(a)")

    for rec, mk, lab in ((r_cst, "o", "Power Law"), (r_zi, "s", "Zero Intelligence")):
        s = np.array(list(rec.spread_hist.keys()))
        p = np.array(list(rec.spread_hist.values()))
        a2.semilogy(s, p, mk + "-", ms=3, label=lab)
    a2.set_xlabel(r"Bid-Ask Spread $s$ (ticks)")
    a2.set_ylabel(r"Fraction of Time $P(s)$")
    s_max = max(max(r_cst.spread_hist), max(r_zi.spread_hist))
    a2.set_xlim(0.5, s_max + 0.5)
    a2.set_ylim(1e-5, 8)
    a2.legend(loc="upper right")
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig1_markov_book", FIG)

    def mean_spread(rec):
        return sum(k * v for k, v in rec.spread_hist.items())

    def spread_batch_se(rec, n_batch=20):
        """Standard error of the time-averaged spread from n_batch batch means.

        The spread after event k holds until event k+1. Batches are long
        compared with the spread's correlation time, so batch means are
        treated as independent.
        """
        s = (rec.best_ask - rec.best_bid)[:-1].astype(float)
        dt = np.diff(rec.t)
        ok = np.abs(s) < 1e6                    # drop instants with an empty side
        means = [np.sum(sb * db) / np.sum(db) for sb, db in
                 zip(np.array_split(s[ok], n_batch), np.array_split(dt[ok], n_batch))]
        return np.std(means, ddof=1) / np.sqrt(n_batch)

    mid = 0.5 * (r_cst.best_bid + r_cst.best_ask)
    write_numbers("e1_markov_book", {
        "EoneNevents": f"{N_EVENTS:,}".replace(",", "{,}"),
        "EoneBurnin": f"{BURN_IN:,}".replace(",", "{,}"),
        "EoneCSTspread": fmt_pm(mean_spread(r_cst), spread_batch_se(r_cst), 3),
        "EoneZIspread": fmt_pm(mean_spread(r_zi), spread_batch_se(r_zi), 3),
        "EoneCSTponetick": fmt(r_cst.spread_hist.get(1, 0.0), 2),
        "EoneZIponetick": fmt(r_zi.spread_hist.get(1, 0.0), 2),
        "EoneCSTduration": fmt(r_cst.t[-1] / 3600, 1),
        "EoneZIduration": fmt(r_zi.t[-1] / 3600, 1),
        "EoneCSTdepthone": fmt(r_cst.depth_profile[0], 2),
        "EoneCSTdepthfar": fmt(r_cst.depth_profile[-1], 2),
        "EoneCSTfarref": fmt(cst.far_depth()[-1], 2),
        "EoneCSTmidrange": fmt(mid.max() - mid.min(), 1),
        "EoneDropped": str(r_cst.dropped_orders + r_zi.dropped_orders),
    })


if __name__ == "__main__":
    main()
