"""E22. Glosten-Milgrom: the spread is the price of adverse selection.

(a) Mean bid-ask spread against the number of trades, for three shares of
    informed traders alpha, with V_H - V_L = 1 (price unit). The first spread
    equals alpha (V_H - V_L) exactly.
(b) The market maker's markout per trade against the true value, by trader
    type: positive against uninformed traders, negative against informed
    ones, and zero in total (zero-profit quotes).

Output: figures/fig22_glosten_milgrom.png, results/e22_glosten_milgrom.{json,tex}
"""
import numpy as np

from common import FIG, fmt, fmt_pm, write_numbers
from stochastic_lob import adverse_selection as gm
from stochastic_lob.plotting import WIDE, panel_label, plt, save, use_style

ALPHAS = (0.1, 0.3, 0.5)
N_TRADES = 60
N_PATHS = 40_000


def main():
    use_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    res = {}
    for a, mk in zip(ALPHAS, ("o", "s", "^")):
        sim = gm.simulate(a, N_TRADES, N_PATHS, rng=220 + int(100 * a))
        sp = sim["spread"].mean(axis=0)
        a1.plot(np.arange(1, N_TRADES + 1), sp, mk + "-", ms=2.5, markevery=5, label=rf"$\alpha = {a}$")
        m = sim["eps"] * (sim["price"] - sim["V"][:, None])
        inf = sim["informed"]
        # Path-level means are independent across paths; standard errors from them.
        per_path_all = m.mean(axis=1)
        per_path_u = np.nanmean(np.where(~inf, m, np.nan), axis=1)
        per_path_i = np.nanmean(np.where(inf, m, np.nan), axis=1)
        res[a] = {
            "spread0": sp[0],
            # first trade at which the mean spread is below half its initial value
            "half_life": (str(int(np.flatnonzero(sp < 0.5 * sp[0])[0]) + 1)
                          if np.any(sp < 0.5 * sp[0]) else rf"\ensuremath{{>}}\,{N_TRADES}"),
            "all": (per_path_all.mean(), per_path_all.std(ddof=1) / np.sqrt(N_PATHS)),
            "u": (np.nanmean(per_path_u), np.nanstd(per_path_u, ddof=1) / np.sqrt(np.sum(np.isfinite(per_path_u)))),
            "i": (np.nanmean(per_path_i), np.nanstd(per_path_i, ddof=1) / np.sqrt(np.sum(np.isfinite(per_path_i)))),
        }
    a1.set_xlabel(r"Trade Number $n$")
    a1.set_ylabel(r"Mean Spread $(a_n - b_n)/(V_H - V_L)$")
    a1.set_ylim(0, 0.75)
    a1.legend(loc="upper right")
    panel_label(a1, "(a)")

    x = np.arange(len(ALPHAS))
    wd = 0.25
    for j, (key, lab) in enumerate((("u", "Uninformed"), ("i", "Informed"), ("all", "All Trades"))):
        a2.bar(x + (j - 1) * wd, [res[a][key][0] for a in ALPHAS], wd,
               yerr=[2 * res[a][key][1] for a in ALPHAS], label=lab, color=f"C{j}")
    a2.axhline(0, color="0.3", lw=0.8)
    a2.set_xticks(x)
    a2.set_xticklabels([rf"$\alpha = {a}$" for a in ALPHAS])
    a2.tick_params(axis="x", which="minor", bottom=False, top=False)
    a2.set_xlabel("Share of Informed Traders")
    a2.set_ylabel(r"Markout per Trade $\varepsilon(p - V)$")
    a2.set_ylim(-0.75, 0.45)
    a2.legend(loc="upper left", ncol=3, fontsize=7)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig22_glosten_milgrom", FIG)

    r3 = res[0.3]
    write_numbers("e22_glosten_milgrom", {
        "EtwentytwoPaths": f"{N_PATHS:,}".replace(",", "{,}"),
        "EtwentytwoSpreadZero": fmt(r3["spread0"], 3),
        "EtwentytwoHalfLife": r3["half_life"],
        "EtwentytwoHalfLifeLow": res[0.1]["half_life"],
        "EtwentytwoHalfLifeHigh": res[0.5]["half_life"],
        "EtwentytwoU": fmt_pm(*r3["u"], 4),
        "EtwentytwoI": fmt_pm(*r3["i"], 4),
        "EtwentytwoAll": fmt_pm(*r3["all"], 4),
    })


if __name__ == "__main__":
    main()
