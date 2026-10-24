"""E6. Propagator model: long-memory signs and the diffusivity condition zeta_c = (1 - gamma_eps)/2.

Output: figures/fig6_propagator.png, results/e6_propagator.{json,tex}
"""
import numpy as np

from common import FIG, fmt, fmt_pm, write_numbers
from stochastic_lob import impact
from stochastic_lob.plotting import WIDE, panel_label, plt, save, use_style

GAMMA_EPS = 0.5    # sign-correlation exponent gamma_eps
N = 2 ** 20        # trades per replicate
N_REP = 8
ZETAS = (0.10, impact.critical_zeta(GAMMA_EPS), 0.40)
FIT_RANGE = (50, 5000)   # lags (trades) used for the log-log slope
RESP_LAGS = (1, 10, 100, 1000)


def main():
    use_style()
    rng = np.random.default_rng(61)
    lags = np.unique(np.geomspace(1, 2e4, 40).astype(int))
    fit_lags = lags[(lags >= FIT_RANGE[0]) & (lags <= FIT_RANGE[1])]
    acf_lags = np.unique(np.geomspace(1, 3000, 30).astype(int))

    slopes = {z: [] for z in ZETAS}
    curves = {z: [] for z in ZETAS}
    acfs = []
    resp = []
    for _ in range(N_REP):
        eps = impact.long_memory_signs(N, GAMMA_EPS, rng)
        acfs.append(impact.autocorrelation(eps, acf_lags))
        for z in ZETAS:
            p = impact.propagator_price(eps, zeta=z)
            curves[z].append(impact.diffusion_signature(p, lags))
            Df = impact.diffusion_signature(p, fit_lags)
            slopes[z].append(np.polyfit(np.log(fit_lags), np.log(Df), 1)[0])
            if z == ZETAS[1]:
                resp.append(impact.response_function(p, eps, RESP_LAGS))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    acf = np.mean(acfs, 0)
    acf_se = np.std(acfs, 0, ddof=1) / np.sqrt(N_REP)
    a1.errorbar(acf_lags, acf, yerr=2 * acf_se, fmt="o", ms=3, mfc="white", label="Simulated Signs")
    a1.plot(acf_lags, impact.sign_autocorrelation_theory(acf_lags, GAMMA_EPS), "-",
            label=r"$(2/\pi)\arcsin\rho_H(\ell)$")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel(r"Lag $\ell$ (trades)")
    a1.set_ylabel(r"Sign Autocorrelation $C_\varepsilon(\ell)$")
    a1.set_ylim(1e-3, 3)
    a1.legend(loc="upper right")
    panel_label(a1, "(a)")

    for z, mk in zip(ZETAS, ("o", "s", "^")):
        D = np.mean(curves[z], 0)
        s = np.array(slopes[z])
        a2.loglog(lags, D, mk + "-", ms=3, mfc="white",
                  label=rf"$\zeta = {z:.2f}$, Slope ${s.mean():+.2f}$")
    a2.axvspan(*FIT_RANGE, color="0.92", zorder=0)
    a2.set_xlabel(r"Lag $\ell$ (trades)")
    a2.set_ylabel(r"Diffusivity $\mathcal{D}(\ell)$ (ticks$^2$)")
    lo, hi = a2.get_ylim()
    a2.set_ylim(lo, hi * 40)
    a2.legend(loc="upper left")
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig6_propagator", FIG)

    resp = np.array(resp)
    num = {
        "EsixGamma": fmt(GAMMA_EPS, 2),
        "EsixNrep": str(N_REP),
        "EsixZetaC": fmt(impact.critical_zeta(GAMMA_EPS), 2),
    }
    for k, (l, key) in enumerate(zip(RESP_LAGS, ("One", "Ten", "Hundred", "Thousand"))):
        num[f"EsixResp{key}"] = fmt_pm(resp[:, k].mean(), resp[:, k].std(ddof=1) / np.sqrt(N_REP), 3)
    for z, key in zip(ZETAS, ("Low", "Crit", "High")):
        s = np.array(slopes[z])
        num[f"EsixSlope{key}"] = fmt_pm(s.mean(), s.std(ddof=1) / np.sqrt(N_REP), 3)
        num[f"EsixTheory{key}"] = fmt(1 - GAMMA_EPS - 2 * z, 2)
    write_numbers("e6_propagator", num)


if __name__ == "__main__":
    main()
