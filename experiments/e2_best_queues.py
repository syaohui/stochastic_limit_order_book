"""E2. Best-queue model (Cont and de Larrard): P(up) and the duration tail.

Output: figures/fig2_best_queues.png, results/e2_best_queues.{json,tex}
"""
import numpy as np

from common import FIG, fmt, write_numbers
from stochastic_lob import best_queues as bq
from stochastic_lob.plotting import WIDE, panel_label, plt, save, use_style

LAM = 1.0          # limit order rate per queue (1/s); balanced case nu = lam
N_MC = 20_000      # Monte Carlo paths per point
T_MAX = 1e5        # simulation cut-off (s)


def main():
    use_style()
    rng = np.random.default_rng(21)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)

    qb_grid = np.arange(1, 13)
    max_dev = 0.0
    for q_a, mk in ((1, "o"), (3, "s"), (6, "^")):
        exact = np.array([bq.p_up_balanced(b, q_a) for b in qb_grid])
        line, = a1.plot(qb_grid, exact, "-", label=rf"$q_a = {q_a}$, Exact")
        a1.plot(qb_grid, bq.p_up_arctan(qb_grid, q_a), ":", color=line.get_color())
        mc, se = [], []
        for b in qb_grid[::2]:
            tau, up = bq.simulate_durations(b, q_a, LAM, LAM, N_MC, T_MAX, rng)
            done = np.isfinite(tau)
            p = up[done].mean()
            mc.append(p)
            se.append(np.sqrt(p * (1 - p) / done.sum()))
        mc, se = np.array(mc), np.array(se)
        max_dev = max(max_dev, np.max(np.abs(mc - exact[::2]) / se))
        a1.errorbar(qb_grid[::2], mc, yerr=2 * se, fmt=mk, ms=3.5, mfc="white",
                    color=line.get_color(), label=rf"$q_a = {q_a}$, Monte Carlo")
    a1.plot([], [], ":", color="0.3", label=r"$(2/\pi)\arctan(q_b/q_a)$")
    a1.set_xlabel(r"Bid Queue Size $q_b$ (orders)")
    a1.set_ylabel(r"Probability of Up Move $p_{\uparrow}$")
    a1.set_xlim(0.5, 12.5)
    a1.set_ylim(0, 1.55)
    a1.legend(loc="upper left", ncol=2, fontsize=7)
    panel_label(a1, "(a)")

    t = np.geomspace(0.1, 1e4, 60)
    tail_err = []
    for (q_b, q_a), mk in (((1, 1), "o"), ((2, 3), "s"), ((5, 5), "^")):
        S = bq.duration_survival(t, q_b, q_a, LAM, LAM)
        line, = a2.loglog(t, S, "-", label=rf"$(q_b, q_a) = ({q_b}, {q_a})$")
        asym = bq.duration_tail_balanced(t, q_b, q_a, LAM)
        a2.loglog(t[asym < 2], asym[asym < 2], "--", color=line.get_color(), lw=0.9)
        tau, _ = bq.simulate_durations(q_b, q_a, LAM, LAM, N_MC, 1.1e4, rng)
        tt = t[::6]
        emp = np.array([(tau > x).mean() for x in tt])
        enough = emp * N_MC >= 20          # at least 20 surviving paths
        a2.loglog(tt[enough], emp[enough], mk, ms=3.5, mfc="white", color=line.get_color())
        tail_err.append(abs(S[-1] / bq.duration_tail_balanced(t[-1], q_b, q_a, LAM) - 1))
    a2.plot([], [], "--", color="0.3", lw=0.9, label=r"$q_b q_a/(\pi\lambda t)$")
    a2.plot([], [], "o", color="0.3", mfc="white", ms=3.5, label="Monte Carlo")
    a2.set_xlabel(r"Time $t$ (s)")
    a2.set_ylabel(r"Survival $P(\tau > t)$")
    a2.set_ylim(1e-4, 5)
    a2.legend(loc="lower left", fontsize=7)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig2_best_queues", FIG)

    write_numbers("e2_best_queues", {
        "EtwoNmc": f"{N_MC:,}".replace(",", "{,}"),
        "EtwoMaxDev": fmt(max_dev, 1),
        "EtwoPupOneThree": fmt(bq.p_up_balanced(1, 3), 3),
        "EtwoPupThreeOne": fmt(bq.p_up_balanced(3, 1), 3),
        "EtwoArctanErrSmall": fmt(abs(bq.p_up_balanced(2, 5) - bq.p_up_arctan(2, 5)), 4),
        "EtwoArctanErr": f"{abs(bq.p_up_balanced(40, 20) - bq.p_up_arctan(40, 20)):.1e}".replace("e-0", r"\times 10^{-") + "}",
        "EtwoTailErr": fmt(100 * max(tail_err), 2),
    })


if __name__ == "__main__":
    main()
