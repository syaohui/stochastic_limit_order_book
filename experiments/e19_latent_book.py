"""E19. Metaorder impact in the latent order book: from linear to square root.

Units: L = 1 share/tick^2, D = 1 tick^2/s, T = 1 s, so u = Q / (L D T) is
the only parameter. (a) Peak impact I(Q) from the self-similar closed form,
with its two limits Q/(L sqrt(pi D T)) and sqrt(2Q/L), and finite-difference
solutions of the PDE. (b) Local exponent d ln I / d ln Q. (c) Price during
and after the metaorder, scaled by its peak, for slow and fast execution,
against the linear-response formula.

Output: figures/fig19_latent_book.png, results/e19_latent_book.{json,tex}
"""
import numpy as np

from common import FIG, fmt, write_numbers
from stochastic_lob import latent_book as lb
from stochastic_lob.plotting import panel_label, plt, save, use_style

U_PDE = (0.03, 0.3, 3.0, 30.0, 100.0)
NX, NT = 16001, 8000


def main():
    use_style()
    Q = np.geomspace(1e-3, 1e3, 80)
    I = np.array([lb.impact_peak(q, 1.0, 1.0, 1.0) for q in Q])
    expo = lb.local_exponent(Q, 1.0, 1.0, 1.0)

    rel_err = []
    pde_I = []
    for u in U_PDE:
        t, y = lb.simulate_metaorder(u, 1.0, n_x=NX, n_t=NT)
        pde_I.append(y[-1])
        rel_err.append(abs(y[-1] / lb.impact_peak(u, 1.0, 1.0, 1.0) - 1))
    # Grid convergence at the largest u: halve the resolution and compare.
    _, y_c = lb.simulate_metaorder(U_PDE[-1], 1.0, n_x=(NX - 1) // 2 + 1, n_t=NT // 2)
    err_coarse = abs(y_c[-1] / lb.impact_peak(U_PDE[-1], 1.0, 1.0, 1.0) - 1)

    paths = {}
    for u in (0.03, 30.0):
        t, y = lb.simulate_metaorder(u, 1.0, t_end=6.0, n_x=NX, n_t=NT)
        paths[u] = (t, y)
    t_lin = np.linspace(0, 6, 600)
    lin = lb.linear_response_path(t_lin, 1.0, 1.0, 1.0, 1.0)
    # Remaining impact at t = 5T, as a share of the peak.
    def remain(u):
        t, y = paths[u]
        return y[np.searchsorted(t, 5.0)] / y[np.searchsorted(t, 1.0)]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.2, 2.5))
    a1.loglog(Q, I, "-", label="Closed Form")
    a1.loglog(Q, Q / np.sqrt(np.pi), "--", color="0.4", lw=0.9, label=r"$Q/(L\sqrt{\pi D T})$")
    a1.loglog(Q, np.sqrt(2 * Q), ":", color="0.2", lw=1.1, label=r"$\sqrt{2Q/L}$")
    a1.loglog(U_PDE, pde_I, "o", ms=4, mfc="white", color="C1", label="PDE")
    a1.set_xlabel(r"Metaorder Size $Q/(LDT)$")
    a1.set_ylabel(r"Peak Impact $I/\sqrt{DT}$")
    a1.set_ylim(3e-4, 3e3)
    a1.legend(loc="upper left", fontsize=7)
    panel_label(a1, "(a)")

    a2.semilogx(Q, expo, "-")
    a2.axhline(1.0, color="0.4", ls="--", lw=0.9)
    a2.axhline(0.5, color="0.2", ls=":", lw=1.1)
    a2.set_xlabel(r"Metaorder Size $Q/(LDT)$")
    a2.set_ylabel(r"Local Exponent $\mathrm{d}\ln I/\mathrm{d}\ln Q$")
    a2.set_ylim(0.3, 1.2)
    panel_label(a2, "(b)")

    for u, mk in ((0.03, "-"), (30.0, "--")):
        t, y = paths[u]
        a3.plot(t, y / y[np.searchsorted(t, 1.0)], mk, label=rf"PDE, $u = {u:g}$")
    a3.plot(t_lin, lin / lin[np.searchsorted(t_lin, 1.0)], ":", color="0.2", lw=1.2,
            label="Linear Response")
    a3.set_xlabel(r"Time $t/T$")
    a3.set_ylabel(r"Impact $I(t)/I(T)$")
    a3.set_ylim(0, 1.45)
    a3.legend(loc="upper right", fontsize=7)
    panel_label(a3, "(c)")
    fig.tight_layout()
    save(fig, "fig19_latent_book", FIG, width_in=7.2)

    # Crossover: Q where the two asymptotes meet, Q/(L D T) = 2 pi.
    write_numbers("e19_latent_book", {
        "EnineteenMaxErr": fmt(100 * max(rel_err), 1),
        "EnineteenErrCoarse": fmt(100 * err_coarse, 1),
        "EnineteenErrFine": fmt(100 * rel_err[-1], 1),
        "EnineteenUmax": f"{U_PDE[-1]:g}",
        "EnineteenExpoLow": fmt(expo[0], 3),
        "EnineteenExpoHigh": fmt(expo[-1], 3),
        "EnineteenRemainSlow": fmt(remain(0.03), 2),
        "EnineteenRemainFast": fmt(remain(30.0), 2),
        "EnineteenRemainLin": fmt(float(np.sqrt(5) - np.sqrt(4)), 2),
    })


if __name__ == "__main__":
    main()
