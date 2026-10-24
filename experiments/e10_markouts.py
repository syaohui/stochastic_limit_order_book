"""E10. Markouts of passive fills in the Markovian book: which fills are toxic.

Output: figures/fig10_markouts.png, results/e10_markouts.{json,tex}
"""
import numpy as np

from common import FIG, fmt, fmt_pm, write_numbers
from stochastic_lob import markouts as mk
from stochastic_lob import markov_lob as ml
from stochastic_lob.plotting import WIDE, panel_label, plt, save, use_style

N_EVENTS = 600_000
HORIZONS = np.array([0.0, 0.5, 1, 2, 5, 10, 20, 40])
H_IMB = 5.0        # horizon (s) for the imbalance panel


def main():
    use_style()
    params = ml.power_law_params(L=30, lam1=1.9, a=0.5, theta0=0.5, mu=1.0)
    rec = ml.simulate_markov_lob(params, N_EVENTS, rng=101, burn_in=20_000)
    f, M = mk.markout_matrix(rec, HORIZONS)
    dep = f["q_before"] == 1

    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    for sel, lab, m_ in ((~dep, "Queue Survives ($q \\geq 2$)", "o"), (dep, "Queue Emptied ($q = 1$)", "s"),
                         (np.ones_like(dep), "All Passive Fills", "^")):
        m, se, n = mk.mean_with_se(M[sel])
        a1.errorbar(HORIZONS, m, yerr=2 * se, fmt=m_ + "-", ms=3.5, mfc="white", label=lab)
    a1.axhline(0, color="0.5", lw=0.6)
    a1.set_xlabel(r"Horizon $h$ (s)")
    a1.set_ylabel(r"Markout $M(h)$ (ticks)")
    a1.set_ylim(-0.5, 1.45)
    a1.legend(loc="upper right", fontsize=7.5)
    panel_label(a1, "(a)")

    # Share of the passive side in the best-level depth: low share means the
    # passive order sat in the thinner queue.
    own = np.where(f["side"] > 0, 1 - f["imb_before"], f["imb_before"])
    j = int(np.flatnonzero(HORIZONS == H_IMB)[0])
    edges = np.quantile(own, np.linspace(0, 1, 6))
    centers, means, ses = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (own >= lo) & (own <= hi)
        m, se, n = mk.mean_with_se(M[sel, j])
        centers.append(np.mean(own[sel])); means.append(m); ses.append(se)
    a2.errorbar(centers, means, yerr=2 * np.array(ses), fmt="o-", ms=3.5, mfc="white",
                label=rf"$h = {H_IMB:.0f}$ s, Quintiles")
    a2.axhline(0, color="0.5", lw=0.6)
    a2.set_xlabel(r"Passive Side Share of Best Depth $\rho$")
    a2.set_ylabel(r"Markout $M(h)$ (ticks)")
    a2.set_xlim(0, 1)
    a2.set_ylim(-0.5, 1.1)
    a2.legend(loc="upper left")
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig10_markouts", FIG)

    m_dep, se_dep, n_dep = mk.mean_with_se(M[dep])
    m_nd, se_nd, n_nd = mk.mean_with_se(M[~dep])
    write_numbers("e10_markouts", {
        "EtenNevents": f"{N_EVENTS:,}".replace(",", "{,}"),
        "EtenNfills": f"{len(f['k']):,}".replace(",", "{,}"),
        "EtenDepShare": fmt(dep.mean(), 2),
        "EtenDepZero": fmt_pm(m_dep[0], se_dep[0], 2),
        "EtenNondepZero": fmt_pm(m_nd[0], se_nd[0], 2),
        "EtenDepTen": fmt_pm(m_dep[5], se_dep[5], 2),
        "EtenNondepTen": fmt_pm(m_nd[5], se_nd[5], 2),
        "EtenLowShare": fmt_pm(means[0], ses[0], 2),
        "EtenHighShare": fmt_pm(means[-1], ses[-1], 2),
    })


if __name__ == "__main__":
    main()
