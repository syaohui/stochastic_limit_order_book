"""E21. Estimating the fill intensity A exp(-k d), and what a wrong k costs.

(a) Quote spells at distances d = 0..15 ticks, each kept until filled or
    for at most 30 s. The Poisson-exposure MLE (censoring handled) is
    compared with the naive fit of fill fractions, which ignores that a
    quote fills at most once per spell.
(b) Model risk. The exact GLFT strategy is computed with a misestimated k
    and run against the true fill process. Loss of certainty equivalent
    (CE, ticks) against the strategy that knows k, with paired paths and
    bootstrap standard errors.

Parameters follow the GLFT example used in E11: sigma = 0.3 tick/sqrt(s),
A = 0.9 1/s, k = 0.3 1/tick, gamma = 0.01 1/tick, T = 600 s, Q = 30.

Output: figures/fig21_fill_estimation.png, results/e21_fill_estimation.{json,tex}
"""
import numpy as np

from common import FIG, fmt, fmt_pm, write_numbers
from stochastic_lob import market_making as mm
from stochastic_lob.plotting import WIDE, panel_label, plt, save, use_style

P = mm.MMParams(S0=0.0, T=600.0, sigma=0.3, dt=0.2, gamma=0.01, k=0.3, A=0.9)
Q = 30
T_MAX = 30.0
N_PER_D = 400
RATIOS = (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0)
N_PATHS = 2000


def main():
    use_style()
    rng = np.random.default_rng(210)
    d = np.repeat(np.arange(0.0, 16.0), N_PER_D)
    tau, filled = mm.simulate_quote_spells(P.A, P.k, d, T_MAX, rng)
    A_hat, k_hat, cov = mm.fit_fill_intensity(d, tau, filled)
    A_nv, k_nv = mm.fit_fill_ratio_naive(d, filled, T_MAX)
    se_lnA, se_k = np.sqrt(np.diag(cov))

    ds = np.unique(d)
    rate = np.array([filled[d == x].sum() / tau[d == x].sum() for x in ds])
    rate_se = np.array([np.sqrt(filled[d == x].sum()) / tau[d == x].sum() for x in ds])
    frac = np.array([filled[d == x].mean() for x in ds]) / T_MAX

    # ------------------------------------------------------------ model risk
    profits = {}
    for r in RATIOS:
        ph = mm.MMParams(**{**P.__dict__, "k": r * P.k})
        strat = mm.glft_strategy(Q, ph)
        profits[r] = mm.simulate(strat, P, n_paths=N_PATHS, rng=211)["profit"]
    ph_nv = mm.MMParams(**{**P.__dict__, "k": k_nv, "A": A_nv})
    profits["naive"] = mm.simulate(mm.glft_strategy(Q, ph_nv), P, n_paths=N_PATHS, rng=211)["profit"]
    ce = {r: mm.certainty_equivalent(v, P.gamma) for r, v in profits.items()}
    boot = {r: [] for r in profits}
    for _ in range(300):
        ix = rng.integers(0, N_PATHS, N_PATHS)
        base = mm.certainty_equivalent(profits[1.0][ix], P.gamma)
        for r in profits:
            boot[r].append(mm.certainty_equivalent(profits[r][ix], P.gamma) - base)
    loss = {r: ce[r] - ce[1.0] for r in profits}
    loss_se = {r: float(np.std(boot[r], ddof=1)) for r in profits}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    dd = np.linspace(0, 15, 200)
    a1.semilogy(dd, P.A * np.exp(-P.k * dd), "-", color="0.3", label=r"True $A e^{-kd}$")
    a1.errorbar(ds, rate, yerr=2 * rate_se, fmt="o", ms=3, mfc="white", label="Fills per Second Exposed")
    a1.semilogy(dd, A_hat * np.exp(-k_hat * dd), "--", label=rf"MLE, $\hat k = {k_hat:.3f}$")
    a1.semilogy(ds, frac, "s", ms=3, label="Fill Fraction / 30 s")
    a1.semilogy(dd, A_nv * np.exp(-k_nv * dd), ":", label=rf"Naive Fit, $\hat k = {k_nv:.3f}$")
    a1.set_xlabel(r"Quote Distance $d$ (ticks)")
    a1.set_ylabel(r"Fill Intensity (s$^{-1}$)")
    a1.set_ylim(3e-3, 30)
    a1.legend(loc="upper right", fontsize=7)
    panel_label(a1, "(a)")

    xs = np.array(RATIOS)
    a2.errorbar(xs, [loss[r] for r in RATIOS], yerr=[2 * loss_se[r] for r in RATIOS], fmt="o-", ms=3.5,
                label="Exact GLFT With $\\hat k$")
    a2.errorbar([k_nv / P.k], [loss["naive"]], yerr=[2 * loss_se["naive"]], fmt="D", ms=4.5, color="C1",
                label=r"Naive $(\hat A, \hat k)$")
    a2.axhline(0, color="0.4", ls=":")
    a2.set_xscale("log")
    a2.set_xticks([0.5, 1, 2])
    a2.set_xticklabels(["0.5", "1", "2"])
    a2.xaxis.set_minor_formatter(plt.NullFormatter())
    a2.set_xlabel(r"Estimation Ratio $\hat k / k$")
    a2.set_ylabel(r"CE Change (ticks)")
    lo = min(min(loss.values()) - 3 * max(loss_se.values()), -10)
    a2.set_ylim(1.15 * lo, -0.35 * lo)
    a2.legend(loc="lower center", fontsize=7)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig21_fill_estimation", FIG)

    write_numbers("e21_fill_estimation", {
        "EtwentyoneSpells": f"{len(d):,}".replace(",", "{,}"),
        "EtwentyoneAhat": fmt_pm(A_hat, A_hat * se_lnA, 3),
        "EtwentyoneKhat": fmt_pm(k_hat, se_k, 3),
        "EtwentyoneAnaive": fmt(A_nv, 3),
        "EtwentyoneKnaive": fmt(k_nv, 3),
        "EtwentyoneCEtrue": fmt(ce[1.0], 1),
        "EtwentyoneLossHalf": fmt_pm(loss[0.5], loss_se[0.5], 1),
        "EtwentyoneLossDouble": fmt_pm(loss[2.0], loss_se[2.0], 1),
        "EtwentyoneLossNaive": fmt_pm(loss["naive"], loss_se["naive"], 1),
        "EtwentyonePaths": f"{N_PATHS:,}".replace(",", "{,}"),
    })


if __name__ == "__main__":
    main()
