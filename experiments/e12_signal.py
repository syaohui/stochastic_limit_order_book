"""E12. When does a short-term signal help a market maker?

The mid drifts at +m0 or -m0 (ticks/s), switching sign at rate kappa (1/s).
The maker sees the sign and quotes the exact GLFT quotes for drift
shrink * m0 * sign. GLFT assumes a permanent drift, so shrink < 1 accounts for
the signal's finite life. Gains are in certainty equivalent (ticks) against
the drift-free GLFT quotes, on common random numbers, with paired bootstrap
standard errors.

Output: figures/fig12_signal.png, results/e12_signal.{json,tex}
"""
import numpy as np

from common import FIG, fmt, fmt_pm, write_numbers
from stochastic_lob import market_making as mm
from stochastic_lob.plotting import SINGLE, plt, save, use_style

P = mm.MMParams(S0=0.0, T=600.0, sigma=0.3, dt=0.2, gamma=0.01, k=0.3, A=0.9)
Q = 30
N_PATHS = 2000
N_BOOT = 200
SHRINKS = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
CASES = [  # (m0 tick/s, mean regime life 1/kappa s)
    (0.02, 300.0, "o"),
    (0.01, 120.0, "s"),
    (0.02, 60.0, "^"),
]


def main():
    use_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    num = {"EtwelveNpaths": f"{N_PATHS:,}".replace(",", "{,}")}
    rng = np.random.default_rng(121)
    for (m0, life, mk), key in zip(CASES, ("Slow", "Mid", "Fast")):
        base = mm.simulate_regime_signal(P, Q, m0, 1 / life, False, N_PATHS, rng=122)["profit"]
        gains, ses = [], []
        for sh in SHRINKS:
            pr = mm.simulate_regime_signal(P, Q, m0, 1 / life, True, N_PATHS, rng=122, shrink=sh)["profit"]
            g = mm.certainty_equivalent(pr, P.gamma) - mm.certainty_equivalent(base, P.gamma)
            boots = []
            for _ in range(N_BOOT):
                i = rng.integers(0, N_PATHS, N_PATHS)
                boots.append(mm.certainty_equivalent(pr[i], P.gamma) - mm.certainty_equivalent(base[i], P.gamma))
            gains.append(g)
            ses.append(np.std(boots, ddof=1))
        gains, ses = np.array(gains), np.array(ses)
        ax.errorbar(SHRINKS, gains, yerr=2 * ses, fmt=mk + "-", ms=3.5, mfc="white",
                    label=rf"$m_0 = {m0}$, Life ${life:.0f}$ s")
        best = int(np.argmax(gains))
        num[f"Etwelve{key}Best"] = fmt(SHRINKS[best], 2)
        num[f"Etwelve{key}BestGain"] = fmt_pm(gains[best], ses[best], 1)
        num[f"Etwelve{key}Raw"] = fmt_pm(gains[-1], ses[-1], 1)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlabel(r"Signal Shrinkage $s$")
    ax.set_ylabel(r"Gain in Certainty Equivalent (ticks)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.9 * (hi - lo))
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig12_signal", FIG)
    write_numbers("e12_signal", num)


if __name__ == "__main__":
    main()
