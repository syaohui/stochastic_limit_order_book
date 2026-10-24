"""E16. Queue-reactive model: estimating intensities, and the price in Model III.

(a) One best queue with HLR-shaped intensities is simulated; the limit,
    cancel and market intensities are re-estimated at each queue size by the
    exposure MLE, with 95% intervals.
(b), (c) The reduced Model III: mean reversion ratio eta = N_c / (2 N_a)
    and the standard deviation of 10-minute price changes, as functions of
    the reinitialization probability theta_reinit, for three values of theta.

Output: figures/fig16_queue_reactive_price.png, results/e16_queue_reactive_price.{json,tex}
"""
import numpy as np

from common import FIG, fmt, write_numbers
from stochastic_lob import queue_reactive as qr
from stochastic_lob.plotting import panel_label, plt, save, use_style

T_ONE = 2e5          # observation time of the single queue (s)
T_M3 = 2e5           # simulated time per Model III run (s)
HORIZON = 600.0      # window for the volatility (s)
THETAS = (0.4, 0.7, 1.0)
REINITS = (0.0, 0.25, 0.5, 0.75, 0.85, 1.0)


def main():
    use_style()
    first, second = qr.hlr_like_rates()

    wait, qb, kind, cens = qr.simulate_queue_events(first, T_ONE, q0=5, rng=161)
    n_max = 20
    est = qr.estimate_intensities(wait, qb, kind, n_max, cens)
    n = np.arange(n_max + 1)
    truth = {"limit": first.limit(n), "cancel": first.cancel(n), "market": first.market(n)}
    z = []
    for name in truth:
        ok = est["count"] >= 30
        ok[0] = ok[0] and name == "limit"
        z.extend(((est[name] - truth[name]) / est["se_" + name])[ok & (est["se_" + name] > 0)])
    max_z = float(np.max(np.abs(z)))

    eta = np.zeros((len(THETAS), len(REINITS)))
    vol = np.zeros_like(eta)
    eta_se = np.zeros_like(eta)
    vol_se = np.zeros_like(eta)
    moves = np.zeros_like(eta)
    for a, th in enumerate(THETAS):
        for b, rr in enumerate(REINITS):
            p = qr.Model3Params(first, second, theta=th, theta_reinit=rr)
            tm, sg, _ = qr.simulate_model3(p, T_M3, rng=1600 + 10 * a + b)
            eta[a, b] = qr.mean_reversion_ratio(sg)
            vol[a, b] = qr.realized_vol(tm, sg, T_M3, HORIZON)
            moves[a, b] = len(sg) / T_M3 * HORIZON
            # Batch-means standard errors over 10 consecutive blocks of moves / time.
            blocks = np.array_split(np.arange(len(sg)), 10)
            e_b = [qr.mean_reversion_ratio(sg[ix]) for ix in blocks]
            eta_se[a, b] = np.std(e_b, ddof=1) / np.sqrt(10)
            edges = np.linspace(0, T_M3, 11)
            v_b = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (tm >= lo) & (tm < hi)
                v_b.append(qr.realized_vol(tm[m] - lo, sg[m], hi - lo, HORIZON))
            vol_se[a, b] = np.std(v_b, ddof=1) / np.sqrt(10)

    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    a1, a2, a3 = axs
    for name, mk in (("limit", "o"), ("cancel", "s"), ("market", "^")):
        line, = a1.plot(n, truth[name], "-", lw=1.0)
        ok = est["count"] >= 30
        a1.errorbar(n[ok], est[name][ok], yerr=2 * est["se_" + name][ok], fmt=mk, ms=3,
                    mfc="white", color=line.get_color(), label=name.capitalize())
    a1.set_xlabel(r"Queue Size $q$ (orders)")
    a1.set_ylabel(r"Intensity (s$^{-1}$)")
    a1.set_ylim(0, 1.9)
    a1.legend(loc="upper right", fontsize=7, ncol=1)
    panel_label(a1, "(a)")

    for a, (th, mk) in enumerate(zip(THETAS, ("o", "s", "^"))):
        a2.errorbar(REINITS, eta[a], yerr=2 * eta_se[a], fmt=mk + "-", ms=3, label=rf"$\theta = {th}$")
        a3.errorbar(REINITS, vol[a], yerr=2 * vol_se[a], fmt=mk + "-", ms=3, label=rf"$\theta = {th}$")
    a2.axhline(0.5, color="0.4", ls=":", lw=0.9)
    a2.set_xlabel(r"Reinitialization $\theta_{\mathrm{reinit}}$")
    a2.set_ylabel(r"Mean Reversion Ratio $\eta$")
    a2.set_ylim(0, 0.85)
    a2.legend(loc="upper left", fontsize=7)
    panel_label(a2, "(b)")
    a3.set_xlabel(r"Reinitialization $\theta_{\mathrm{reinit}}$")
    a3.set_ylabel(r"10-Minute Volatility (ticks)")
    a3.set_ylim(0, 1.5 * vol.max())
    a3.legend(loc="upper left", fontsize=7)
    panel_label(a3, "(c)")
    fig.tight_layout()
    save(fig, "fig16_queue_reactive_price", FIG, width_in=7.2)

    i7 = THETAS.index(0.7)
    j0, j85, j1 = REINITS.index(0.0), REINITS.index(0.85), REINITS.index(1.0)
    write_numbers("e16_queue_reactive_price", {
        "EsixteenEvents": f"{len(wait):,}".replace(",", "{,}"),
        "EsixteenMaxZ": fmt(max_z, 1),
        "EsixteenEtaMech": fmt(eta[THETAS.index(1.0), j0], 2),
        "EsixteenEtaSevenZero": fmt(eta[i7, j0], 2),
        "EsixteenEtaSevenHLR": fmt(eta[i7, j85], 2),
        "EsixteenEtaSevenOne": fmt(eta[i7, j1], 2),
        "EsixteenVolMech": fmt(vol[THETAS.index(1.0), j0], 2),
        "EsixteenVolSevenZero": fmt(vol[i7, j0], 2),
        "EsixteenVolSevenHLR": fmt(vol[i7, j85], 2),
        "EsixteenMovesSevenZero": fmt(moves[i7, j0], 0),
        "EsixteenMovesSevenHLR": fmt(moves[i7, j85], 0),
        "EsixteenEtaSE": fmt(np.max(eta_se), 2),
    })


if __name__ == "__main__":
    main()
