"""E3. Queue-reactive model, Model I: rates and the stationary queue law.

Output: figures/fig3_queue_reactive.png, results/e3_queue_reactive.{json,tex}
"""
import numpy as np

from common import FIG, fmt, write_numbers
from stochastic_lob import queue_reactive as qr
from stochastic_lob.plotting import WIDE, panel_label, plt, save, use_style

T_SIM = 2e5        # simulated time (s)
Q_MAX = 60


def main():
    use_style()
    rates = qr.illustrative_rates()
    q = np.arange(0, 21)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    a1.plot(q, rates.limit(q), "o-", ms=3, label=r"Limit $\lambda^{L}(q)$")
    a1.plot(q, rates.cancel(q), "s-", ms=3, label=r"Cancel $\lambda^{C}(q)$")
    a1.plot(q, rates.market(q), "^-", ms=3, label=r"Market $\lambda^{M}(q)$")
    a1.set_xlabel(r"Queue Size $q$ (orders)")
    a1.set_ylabel(r"Intensity (s$^{-1}$)")
    a1.set_ylim(0, 4.2)
    a1.legend(loc="upper left")
    panel_label(a1, "(a)")

    pi = qr.stationary_distribution(rates, Q_MAX)
    occ, n_ev = qr.simulate_queue(rates, T_SIM, rng=31)
    qq = np.arange(Q_MAX + 1)
    a2.semilogy(qq, pi, "-", label="Product Formula")
    a2.semilogy(qq, occ[: Q_MAX + 1], "o", ms=3, mfc="white", label="Simulation")
    a2.set_xlabel(r"Queue Size $q$ (orders)")
    a2.set_ylabel(r"Stationary Probability $\pi(q)$")
    a2.set_xlim(-0.5, 15.5)
    a2.set_ylim(1e-6, 3)
    a2.legend(loc="upper right")
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig3_queue_reactive", FIG)

    tv = 0.5 * np.abs(occ[: Q_MAX + 1] - pi).sum()
    write_numbers("e3_queue_reactive", {
        "EthreeTsim": f"{T_SIM:.0e}".replace("e+0", "\\times 10^{") + "}",
        "EthreeNevents": f"{n_ev:,}".replace(",", "{,}"),
        "EthreeTV": fmt(tv, 4),
        "EthreeMeanQ": fmt(qr.mean_queue_size(pi), 2),
        "EthreeModeQ": str(int(np.argmax(pi))),
    })


if __name__ == "__main__":
    main()
