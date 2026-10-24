"""E13. Liquidity provision in an AMM: loss-versus-rebalancing, fees, and concentration.

Time unit: one day. sigma = 0.04 per sqrt(day), about 76% annualized over 365 days.

Output: figures/fig13_lvr.png, results/e13_lvr.{json,tex}
"""
import numpy as np

from common import FIG, fmt, fmt_pm, write_numbers
from stochastic_lob import amm
from stochastic_lob.plotting import panel_label, plt, save, use_style

SIGMA = 0.04
N_PATHS = 4000
FEES = np.array([0.0, 1e-4, 5e-4, 1e-3, 3e-3, 1e-2])
BLOCKS = [(12.0, "o"), (2.0, "s"), (0.25, "^")]      # block time (s)


def main():
    use_style()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.2, 2.6))

    # (a) Cumulative LVR against the exact expectation.
    T = 30.0
    times = np.linspace(0, T, 31)
    lvr_paths = []
    rng = np.random.default_rng(131)
    for Tk in times[1:]:
        lvr_paths.append(amm.simulate_lvr_cpmm(SIGMA, Tk, int(Tk * 48), 2000, rng=rng))
    m = np.array([0.0] + [x.mean() for x in lvr_paths])
    se = np.array([0.0] + [x.std(ddof=1) / np.sqrt(x.size) for x in lvr_paths])
    V0 = amm.cpmm_value(1.0)
    a1.plot(times, 100 * amm.expected_lvr_cpmm(times, SIGMA) / V0, "-", label=r"$V_0(1 - e^{-\sigma^2 t/8})$")
    a1.errorbar(times[::3], 100 * m[::3] / V0, yerr=200 * se[::3] / V0, fmt="o", ms=3, mfc="white",
                label="Simulation")
    a1.set_xlabel(r"Time $t$ (days)")
    a1.set_ylabel(r"$\mathbb{E}[\mathrm{LVR}_t] / V_0$ (%)")
    a1.set_ylim(0, 1.0)
    a1.legend(loc="upper left", fontsize=7)
    panel_label(a1, "(a)")

    # (b) Arbitrage profit and fee income as fractions of the no-fee LVR.
    day_lvr = amm.expected_lvr_cpmm(1.0, SIGMA)
    fee_share = {}
    for blk, mk in BLOCKS:
        arb, fee = [], []
        for f in FEES:
            r = amm.simulate_cpmm_arbitrage(SIGMA, 1.0, blk / 86400, f, N_PATHS // 4, rng=132)
            arb.append(r["arb"].mean() / day_lvr)
            fee.append(r["fees"].mean() / day_lvr)
        x = np.maximum(FEES * 1e4, 0.3)
        a2.semilogx(x, arb, mk + "-", ms=3, label=rf"Arbitrage, {blk:g} s Blocks")
        a2.semilogx(x, fee, mk + "--", ms=3, mfc="white", color=a2.lines[-1].get_color())
        fee_share[blk] = (arb, fee)
    a2.plot([], [], "--", color="0.3", label="Fee Income (Dashed)")
    a2.set_xlabel(r"Pool Fee $f$ (bp)")
    a2.set_ylabel(r"Share of No-Fee LVR")
    a2.set_ylim(0, 1.75)
    a2.legend(loc="upper left", fontsize=6.5)
    panel_label(a2, "(b)")

    # (c) LVR per unit capital for a centred range [P/w, P w].
    w = np.geomspace(1.01, 100, 60)
    a3.loglog(w, amm.lvr_per_capital_concentrated(SIGMA, w) * 365 * 100, "-", label="Formula")
    ws = np.array([1.05, 1.2, 2.0, 5.0])
    sim = []
    for wi in ws:
        lvr, _ = amm.simulate_lvr_range(SIGMA, 0.1, 200, 4000, 1 / wi, wi, rng=133)
        V0r = amm.range_position(1.0, 1 / wi, wi)[2]
        sim.append(lvr.mean() / V0r / 0.1 * 365 * 100)
    a3.loglog(ws, sim, "o", ms=3.5, mfc="white", label="Simulation, 0.1 Day")
    a3.axhline(SIGMA ** 2 / 8 * 365 * 100, color="0.5", lw=0.6, ls=":")
    a3.set_xlabel(r"Range Width $w$")
    a3.set_ylabel(r"LVR per Capital (% per year)")
    a3.set_ylim(5, 2e4)
    a3.legend(loc="upper right", fontsize=7)
    panel_label(a3, "(c)")
    fig.tight_layout()
    save(fig, "fig13_lvr", FIG)

    j5 = int(np.flatnonzero(FEES == 5e-4)[0])
    j30 = int(np.flatnonzero(FEES == 3e-3)[0])
    write_numbers("e13_lvr", {
        "EthirteenSigma": fmt(SIGMA, 2),
        "EthirteenAnnualLVR": fmt(SIGMA ** 2 / 8 * 365 * 100, 1),
        "EthirteenSimThirty": fmt_pm(100 * m[-1] / V0, 100 * se[-1] / V0, 3),
        "EthirteenExactThirty": fmt(100 * amm.expected_lvr_cpmm(T, SIGMA) / V0, 3),
        "EthirteenArbFiveTwelve": fmt(fee_share[12.0][0][j5], 2),
        "EthirteenFeeFiveTwelve": fmt(fee_share[12.0][1][j5], 2),
        "EthirteenArbThirtyTwelve": fmt(fee_share[12.0][0][j30], 2),
        "EthirteenFeeThirtyTwelve": fmt(fee_share[12.0][1][j30], 2),
        "EthirteenArbThirtyFast": fmt(fee_share[0.25][0][j30], 2),
        "EthirteenAmpTen": fmt(1 / (1 - 1.1 ** -0.5), 1),
    })


if __name__ == "__main__":
    main()
