"""E9. Value of queue position: fill probability and expected markout by place in the queue.

Output: figures/fig9_queue_value.png, results/e9_queue_value.{json,tex}
"""
import numpy as np

from common import FIG, fmt, write_numbers
from stochastic_lob import queue_position as qp
from stochastic_lob.plotting import WIDE, panel_label, plt, save, use_style

Q_TOTAL = 16       # orders in the ask queue including ours
C0 = 10            # orders in the bid queue
BASE = dict(lam=1.0, mu=0.9, theta=0.01, jump=2.0)
NUS = (0.0, 0.03)  # informed sweep rate per side (1/s)


def main():
    use_style()
    a = np.arange(Q_TOTAL)
    b = Q_TOTAL - 1 - a
    res = {}
    for nu in NUS:
        V = qp.solve_queue_value(qp.QueueRates(nu=nu, **BASE), a_max=Q_TOTAL - 1, b_max=50, c_max=50)
        res[nu] = (V.p_fill[a, b, C0], V.value[a, b, C0], V.adverse[a, b, C0])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    for nu, mk in zip(NUS, ("o", "s")):
        lab = "No Informed Flow" if nu == 0 else rf"Sweeps, $\nu = {nu}$ s$^{{-1}}$"
        a1.plot(a, res[nu][0], mk + "-", ms=3.5, label=lab)
    a1.set_xlabel(r"Orders Ahead in Queue $a$")
    a1.set_ylabel(r"Fill Probability $P_{\mathrm{fill}}$")
    a1.set_ylim(0.5, 1.12)
    a1.set_xticks(range(0, Q_TOTAL, 3))
    a1.legend(loc="lower left")
    panel_label(a1, "(a)")

    for nu, mk in zip(NUS, ("o", "s")):
        p, v, adv = res[nu]
        lab = "No Informed Flow" if nu == 0 else rf"Sweeps, $\nu = {nu}$ s$^{{-1}}$"
        a2.plot(a, v / p, mk + "-", ms=3.5, label=lab)
    a2.axhline(0, color="0.5", lw=0.6)
    a2.axhline(0.5, color="0.5", lw=0.6, ls=":")
    a2.set_xlabel(r"Orders Ahead in Queue $a$")
    a2.set_ylabel(r"Markout per Fill $\mathbb{E}[M \mid \mathrm{fill}]$ (ticks)")
    a2.set_ylim(-0.6, 1.0)
    a2.set_xticks(range(0, Q_TOTAL, 3))
    a2.legend(loc="lower left")
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig9_queue_value", FIG)

    p, v, adv = res[NUS[1]]
    per_fill = v / p
    neg = np.flatnonzero(per_fill < 0)
    p0, v0, _ = res[0.0]
    write_numbers("e9_queue_value", {
        "EnineQtot": str(Q_TOTAL),
        "EnineCzero": str(C0),
        "EnineNu": fmt(NUS[1], 2),
        "EnineFrontPerFill": fmt(per_fill[0], 2),
        "EnineBackPerFill": fmt(per_fill[-1], 2),
        "EnineFrontFill": fmt(p[0], 2),
        "EnineBackFill": fmt(p[-1], 2),
        "EnineBreakEven": str(int(neg[0])) if neg.size else "none",
        "EnineZIFront": fmt((v0 / p0)[0], 2),
        "EnineZIBack": fmt((v0 / p0)[-1], 2),
    })


if __name__ == "__main__":
    main()
