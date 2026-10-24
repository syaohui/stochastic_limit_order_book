"""E14. Order flow beyond the best quotes: multi-level and integrated OFI.

Output: figures/fig14_mlofi.png, results/e14_mlofi.{json,tex}
"""
import numpy as np

from common import FIG, fmt, write_numbers
from stochastic_lob import impact
from stochastic_lob import markov_lob as ml
from stochastic_lob.plotting import WIDE, panel_label, plt, save, use_style

N_EVENTS = 600_000
K = 5              # price levels per side
WINDOWS = (2.0, 10.0)   # aggregation windows (s)


def main():
    use_style()
    params = ml.power_law_params(L=30, lam1=1.9, a=0.5, theta0=0.5, mu=1.0)
    rec = ml.simulate_markov_lob(params, N_EVENTS, rng=141, burn_in=20_000, record_levels=K)
    ok = (rec.q_bid > 0) & (rec.q_ask > 0)
    t, bp, ap = rec.t[ok], rec.best_bid[ok], rec.best_ask[ok]
    bd, ad = rec.bid_depth[ok], rec.ask_depth[ok]
    mean_depth = 0.5 * (bd.mean() + ad.mean())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    num = {"EfourteenK": str(K)}
    for dt, mk, key in zip(WINDOWS, ("o", "s"), ("Short", "Long")):
        ofi, dmid = impact.aggregate_multilevel_ofi(t, bp, bd, ap, ad, dt)
        r2_in, r2_out = [], []
        for m in range(1, K + 1):
            r2_in.append(impact.ols_multi(ofi[:, :m], dmid)["r2"])
            r2_out.append(impact.out_of_sample_r2(ofi[:, :m], dmid))
        w, iofi = impact.integrated_ofi(ofi, mean_depth)
        n_tr = len(dmid) // 2
        r2_int = impact.out_of_sample_r2(iofi[:, None], dmid)
        a1.plot(range(1, K + 1), r2_out, mk + "-", ms=3.5, label=rf"$\Delta t = {dt:g}$ s, Out of Sample")
        a1.plot(range(1, K + 1), r2_in, mk + "--", ms=3.5, mfc="white", color=a1.lines[-1].get_color(),
                label=rf"$\Delta t = {dt:g}$ s, In Sample")
        a1.axhline(r2_int, color=a1.lines[-1].get_color(), lw=0.8, ls=":")
        if dt == WINDOWS[1]:
            fit = impact.ols_multi(ofi, dmid)
            coef, se = fit["coef"][1:], fit["se"][1:]
            a2.errorbar(np.arange(K), coef, yerr=2 * se, fmt="o-", ms=3.5, mfc="white", label="OLS Coefficient")
            a2.set_xticks(np.arange(K))
            num["EfourteenCoefZero"] = fmt(coef[0], 3)
            num["EfourteenCoefOne"] = fmt(coef[1], 3)
            num["EfourteenWeights"] = ", ".join(f"{x:.2f}" for x in w)
        num[f"Efourteen{key}One"] = fmt(r2_out[0], 2)
        num[f"Efourteen{key}All"] = fmt(r2_out[-1], 2)
        num[f"Efourteen{key}Int"] = fmt(r2_int, 2)
        num[f"Efourteen{key}N"] = f"{len(dmid):,}".replace(",", "{,}")
    a1.set_xlabel(r"Levels Included $m$")
    a1.set_ylabel(r"Explained Variance $R^2$")
    a1.set_ylim(0, 1.25)
    a1.legend(loc="upper left", fontsize=6.5, ncol=1)
    panel_label(a1, "(a)")
    a2.axhline(0, color="0.5", lw=0.6)
    a2.set_xlabel(r"Price Level $m$ (ticks from best)")
    a2.set_ylabel(r"Impact Coefficient (ticks per order)")
    lo, hi = a2.get_ylim()
    a2.set_ylim(min(lo, -0.02), hi + 0.6 * (hi - lo))
    a2.legend(loc="upper right")
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig14_mlofi", FIG)
    write_numbers("e14_mlofi", num)


if __name__ == "__main__":
    main()
