"""E18. Estimating the propagator from trades, and why the response function is not it.

Trade signs with long memory (gamma_eps = 0.5) drive prices through the
propagator G(l) = l^(-zeta) with the critical zeta = 0.25. Each price change
also carries independent noise of one tick. G is re-estimated by solving the
Toeplitz system of sign correlations, with truncation lag K = 100 and
K = 2000, and 95% intervals from a block bootstrap.

Output: figures/fig18_propagator_estimation.png, results/e18_propagator_estimation.{json,tex}
"""
import numpy as np

from common import FIG, fmt, fmt_pm, write_numbers
from stochastic_lob import impact as im
from stochastic_lob.plotting import WIDE, panel_label, plt, save, use_style

N = 2 ** 20
GAMMA = 0.5
ZETA = 0.25
NOISE = 1.0          # standard deviation of the extra price noise per trade (ticks)
LAGS_FIT = (2, 100)  # lag range for the log-log fit of zeta


def zeta_fit(G):
    l = np.arange(LAGS_FIT[0], LAGS_FIT[1] + 1)
    slope, _ = np.polyfit(np.log(l), np.log(G[l - 1]), 1)
    return -slope


def main():
    use_style()
    rng = np.random.default_rng(180)
    eps = im.long_memory_signs(N + 1, GAMMA, rng)
    P = im.propagator_price(eps, 1.0, ZETA)
    dP = np.diff(P) + NOISE * rng.standard_normal(N)
    eps = eps[:N]

    res = {}
    for K in (100, 2000):
        G = im.estimate_propagator(eps, dP, K)
        se = im.block_bootstrap(lambda ix: im.estimate_propagator(eps[ix], dP[ix], K), N,
                                n_blocks=32, n_boot=60, rng=181)
        zs = im.block_bootstrap(lambda ix: zeta_fit(im.estimate_propagator(eps[ix], dP[ix], K)), N,
                                n_blocks=32, n_boot=60, rng=182)
        res[K] = (G, se, zeta_fit(G), float(zs))

    lags = np.unique(np.geomspace(1, 2000, 60).astype(int))
    Pn = np.cumsum(np.concatenate([[0.0], dP]))       # price before each trade, with noise
    R = im.response_function(Pn[:N], eps, lags)
    l_all = np.arange(1, 2001)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    a1.loglog(l_all, l_all ** -ZETA, "-", color="0.3", label=r"True $G(\ell) = \ell^{-1/4}$")
    for K, mk in ((100, "o"), (2000, "s")):
        G, se, _, _ = res[K]
        pick = lags[lags <= K]
        a1.errorbar(pick, G[pick - 1], yerr=2 * se[pick - 1], fmt=mk, ms=3, mfc="white",
                    label=rf"Toeplitz Estimate, $K = {K}$")
    a1.set_xlabel(r"Lag $\ell$ (trades)")
    a1.set_ylabel(r"Propagator $G(\ell)$ (ticks)")
    a1.set_ylim(0.08, 3)
    a1.legend(loc="upper right", fontsize=7.5)
    panel_label(a1, "(a)")

    a2.semilogx(lags, R, "o-", ms=3, label=r"Response $\mathcal{R}(\ell)$")
    a2.semilogx(l_all, l_all ** -ZETA, "-", color="0.3", label=r"Propagator $G(\ell)$")
    a2.set_xlabel(r"Lag $\ell$ (trades)")
    a2.set_ylabel(r"Price Response (ticks)")
    a2.set_ylim(0, 1.25 * R.max())
    a2.legend(loc="upper left", fontsize=7.5)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig18_propagator_estimation", FIG)

    G1, se1, z1, zs1 = res[100]
    G2, se2, z2, zs2 = res[2000]
    write_numbers("e18_propagator_estimation", {
        "EeighteenN": r"2^{20}",
        "EeighteenGhundredShort": fmt(G1[99], 3),
        "EeighteenGhundredLong": fmt_pm(G2[99], se2[99], 3),
        "EeighteenGtrue": fmt(100 ** -ZETA, 3),
        "EeighteenZetaShort": fmt_pm(z1, zs1, 3),
        "EeighteenZetaLong": fmt_pm(z2, zs2, 3),
        "EeighteenRone": fmt(R[0], 2),
        "EeighteenRmax": fmt(R.max(), 2),
    })


if __name__ == "__main__":
    main()
