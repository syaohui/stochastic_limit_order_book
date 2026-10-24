"""E23. The Cont-de Larrard formula applied to a book it does not describe.

The power-law Markovian book is simulated. While the spread is one tick, its
best queues grow at lam(1) and shrink at mu + theta q: depletion speeds up
with queue size, unlike the constant rates of Cont and de Larrard (CdL).
A practitioner fits CdL by measuring lam and nu = (market orders +
cancellations at the best) / time, then predicts P(next move up | q_b, q_a).

(a) Empirical P(up) against q_b for three q_a, the fitted CdL prediction,
    and the exact answer for the true state-dependent rates.
(b) All cells with at least 300 observations: CdL and exact predictions
    against the empirical frequency.

Standard errors are clustered by price-move spell (see empirical_p_up):
samples taken between the same two mid moves share one outcome.

Output: figures/fig23_cdl_on_book.png, results/e23_cdl_on_book.{json,tex}
"""
import numpy as np

from common import FIG, fmt, write_numbers
from stochastic_lob import best_queues as bq, markov_lob as ml, queue_position as qp
from stochastic_lob.plotting import WIDE, panel_label, plt, save, use_style

N_EVENTS = 1_500_000
MAX_Q = 10
MIN_N = 300


def main():
    use_style()
    p = ml.power_law_params(L=30, lam1=1.9, a=0.5, theta0=0.5, mu=1.0)
    rec = ml.simulate_markov_lob(p, N_EVENTS, rng=230, burn_in=20_000)
    P, Nn, SE = ml.empirical_p_up(rec, MAX_Q)

    # CdL fit from one-tick-spread periods: per-queue rates.
    s1 = (rec.best_ask - rec.best_bid)[:-1] == 1
    dt = np.diff(rec.t)
    T1 = float(np.sum(dt[s1]))
    ev = rec.event[1:][s1]
    px = rec.event_price[1:][s1]
    bid0, ask0 = rec.best_bid[:-1][s1], rec.best_ask[:-1][s1]
    own_best = np.where(np.isin(ev, (0, 4)), bid0, ask0)      # best quote on the order's side
    joins = np.isin(ev, (0, 1)) & (px == own_best)             # limit orders joining a best queue
    leaves = np.isin(ev, (2, 3)) | (np.isin(ev, (4, 5)) & (px == own_best))
    lam_hat = float(np.sum(joins)) / (2 * T1)                 # per queue (1/s)
    nu_hat = float(np.sum(leaves)) / (2 * T1)                 # market orders + cancels at the best

    h = qp.p_up_state_dependent(qp.QueueRates(lam=p.lam[0], mu=p.mu, theta=p.theta[0]), 60, 60)
    qb = np.arange(1, MAX_Q + 1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    for q_a, mk in ((1, "o"), (3, "s"), (6, "^")):
        ok = Nn[1:, q_a] >= MIN_N
        e = P[1:, q_a]
        se = SE[1:, q_a]
        cdl = np.array([bq.p_up(b, q_a, lam_hat, nu_hat) for b in qb])
        line, = a1.plot(qb, cdl, "--", lw=1.0)
        a1.plot(qb, h[1:MAX_Q + 1, q_a], "-", lw=1.0, color=line.get_color())
        a1.errorbar(qb[ok], e[ok], yerr=2 * se[ok], fmt=mk, ms=3, mfc="white", color=line.get_color(),
                    label=rf"$q_a = {q_a}$")
    a1.plot([], [], "--", color="0.4", lw=1.0, label="CdL Fit")
    a1.plot([], [], "-", color="0.4", lw=1.0, label="Exact, True Rates")
    a1.set_xlabel(r"Bid Queue Size $q_b$ (orders)")
    a1.set_ylabel(r"Probability of Up Move $p_\uparrow$")
    a1.set_ylim(0, 1.45)
    a1.legend(loc="upper left", ncol=2, fontsize=7)
    panel_label(a1, "(a)")

    emp, pc, pe, ses = [], [], [], []
    for b in range(1, MAX_Q + 1):
        for a in range(1, MAX_Q + 1):
            if Nn[b, a] >= MIN_N:
                emp.append(P[b, a])
                ses.append(SE[b, a])
                pc.append(bq.p_up(b, a, lam_hat, nu_hat))
                pe.append(h[b, a])
    emp, pc, pe, ses = map(np.array, (emp, pc, pe, ses))
    a2.plot([0, 1], [0, 1], ":", color="0.4")
    a2.plot(emp, pc, "o", ms=3, mfc="white", label="CdL Fit")
    a2.plot(emp, pe, "s", ms=2.5, label="Exact, True Rates")
    a2.set_xlabel(r"Empirical $p_\uparrow$")
    a2.set_ylabel(r"Predicted $p_\uparrow$")
    a2.set_xlim(0, 1)
    a2.set_ylim(0, 1.3)
    a2.legend(loc="upper left", fontsize=7.5)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig23_cdl_on_book", FIG)

    write_numbers("e23_cdl_on_book", {
        "EtwentythreeLam": fmt(lam_hat, 3),
        "EtwentythreeNu": fmt(nu_hat, 3),
        "EtwentythreeCells": str(len(emp)),
        "EtwentythreeMaeCdl": fmt(np.mean(np.abs(pc - emp)), 3),
        "EtwentythreeMaeExact": fmt(np.mean(np.abs(pe - emp)), 3),
        "EtwentythreeMaxZExact": fmt(np.max(np.abs(pe - emp) / ses), 1),
        "EtwentythreeMaxErrCdl": fmt(np.max(np.abs(pc - emp)), 2),
        "EtwentythreeShareOne": fmt(np.mean(s1), 2),
        "EtwentythreeMoves": f"{int(np.sum(np.diff(rec.best_bid + rec.best_ask) != 0)):,}".replace(",", "{,}"),
        "EtwentythreeNevents": f"{N_EVENTS:,}".replace(",", "{,}"),
    })


if __name__ == "__main__":
    main()
