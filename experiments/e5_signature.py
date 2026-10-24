"""E5. Hawkes price model: realized-variance signature plot, theory against simulation.

Output: figures/fig5_signature.png, results/e5_signature.{json,tex}
"""
import numpy as np

from common import FIG, fmt, write_numbers
from stochastic_lob import hawkes
from stochastic_lob.plotting import SINGLE, plt, save, use_style

MU = 0.5           # baseline rate of up and of down ticks (1/s)
BETA = 1.0         # kernel decay rate (1/s)
T = 5_000.0        # length of each replicate (s)
N_REP = 10         # independent replicates per parameter set
CASES = [          # (alpha_s, alpha_c) in 1/s
    (0.6, 0.1, "Self-Exciting", "o"),
    (0.0, 0.0, "Poisson", "s"),
    (0.1, 0.6, "Cross-Exciting", "^"),
]


def main():
    use_style()
    rng = np.random.default_rng(51)
    taus = np.geomspace(0.05, 100, 14)
    tt = np.geomspace(0.03, 150, 200)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    worst = 0.0
    for a_s, a_c, lab, mk in CASES:
        th = hawkes.signature_plot_theory(tt, MU, a_s, a_c, BETA)
        line, = ax.semilogx(tt, th, "-", label=lab)
        m = hawkes.ExpHawkes([MU, MU], [[a_s, a_c], [a_c, a_s]], [[BETA, BETA], [BETA, BETA]])
        reps = []
        for _ in range(N_REP):
            ts, ty = m.simulate(T, rng)
            reps.append(hawkes.realized_signature(ts, ty, T, taus))
        reps = np.array(reps)
        mean = reps.mean(0)
        se = reps.std(0, ddof=1) / np.sqrt(N_REP)
        ax.errorbar(taus, mean, yerr=2 * se, fmt=mk, ms=3, mfc="white", color=line.get_color())
        z = np.abs(mean - hawkes.signature_plot_theory(taus, MU, a_s, a_c, BETA)) / se
        worst = max(worst, float(np.max(z)))
    ax.plot([], [], "o", color="0.3", mfc="white", ms=3, label="Simulation")
    ax.set_xlabel(r"Sampling Interval $h$ (s)")
    ax.set_ylabel(r"Signature $C(h)$ (ticks$^2$ s$^{-1}$)")
    ax.set_ylim(0, 1.3 * ax.get_ylim()[1])
    ax.legend(loc="upper left", ncol=1)
    fig.tight_layout()
    save(fig, "fig5_signature", FIG)

    write_numbers("e5_signature", {
        "EfiveNrep": str(N_REP),
        "EfiveT": f"{int(T):,}".replace(",", "{,}"),
        "EfiveMaxZ": fmt(worst, 1),
        "EfiveCrossZeroLim": fmt(hawkes.signature_plot_theory(1e-9, MU, 0.1, 0.6, BETA), 3),
        "EfiveCrossInfLim": fmt(hawkes.signature_plot_theory(1e9, MU, 0.1, 0.6, BETA), 3),
    })


if __name__ == "__main__":
    main()
