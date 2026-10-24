"""E20. The micro-price on a simulated book.

The Markovian (CST-type) book is simulated; the first half of the events
trains Stoikov's micro-price (imbalance in 10 buckets, spread 1 or 2 ticks)
and the second half tests it.

(a) Expected mid change after the next k mid changes, by imbalance, at a
    one-tick spread: model g_1 + ... + g_k (lines, training half) against
    realized means on the test half (points, block standard errors), and the
    weighted mid (I - 1/2) s.
(b) Mean squared error of each predictor of the future mid, relative to the
    mid itself, against the horizon. Errors are batch means over 20 blocks.

Output: figures/fig20_microprice.png, results/e20_microprice.{json,tex}
"""
import numpy as np

from common import FIG, fmt, fmt_pm, write_numbers
from stochastic_lob import markov_lob as ml, microprice as mp
from stochastic_lob.plotting import WIDE, panel_label, plt, save, use_style

N_EVENTS = 600_000
N_IMB = 10
HORIZONS = (1.0, 3.0, 10.0, 30.0, 100.0)


def main():
    use_style()
    p = ml.power_law_params(L=30, lam1=1.9, a=0.5, theta0=0.5, mu=1.0)
    rec = ml.simulate_markov_lob(p, N_EVENTS, rng=200, burn_in=20_000)
    mid = 0.5 * (rec.best_bid + rec.best_ask).astype(float)
    spr = (rec.best_ask - rec.best_bid).astype(float)
    idx, imb = mp.states(rec.q_bid, rec.q_ask, spr, N_IMB, 2)
    n = len(mid)
    h = n // 2
    fit = mp.fit(idx[:h], mid[:h], N_IMB, 2)
    G = fit["G"]
    m0, wmid, micro = mp.predictors(idx, imb, mid, spr, G)

    centres = (np.arange(N_IMB) + 0.5) / N_IMB
    test = np.zeros(n, dtype=bool)
    test[h:] = True

    def future(H):
        j = np.searchsorted(rec.t, rec.t + H)
        ok = j < n
        return mid[np.minimum(j, n - 1)], ok

    # Realized mid after the k-th mid change following each test observation.
    moves = np.flatnonzero(np.diff(mid) != 0) + 1
    nxt = np.searchsorted(moves, np.arange(n), side="right")

    def after_k(k):
        j = nxt + k - 1
        ok = j < len(moves)
        return mid[moves[np.minimum(j, len(moves) - 1)]], ok

    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    cum = np.cumsum(fit["g"], axis=0)
    real = {}
    for k, mk in ((1, "v"), (3, "s"), (10, "^")):
        line, = a1.plot(centres, cum[k - 1][:N_IMB], "-", lw=1.0)
        tgt, ok = after_k(k)
        sel = np.flatnonzero(test & ok & (idx >= 0) & (idx < N_IMB))
        d = (tgt - mid)[sel]
        b = idx[sel]
        blocks = np.array_split(np.arange(len(sel)), 20)
        means = np.array([d[b == q].mean() for q in range(N_IMB)])
        bm = np.array([[d[bl][b[bl] == q].mean() for q in range(N_IMB)] for bl in blocks])
        ses = np.nanstd(bm, axis=0, ddof=1) / np.sqrt(20)
        real[k] = (means, ses)
        a1.errorbar(centres, means, yerr=2 * ses, fmt=mk, ms=3, mfc="white", color=line.get_color(),
                    label=rf"$k = {k}$")
    a1.plot([], [], "-", color="0.5", lw=1.0, label=r"Model $\sum_{i \leq k} g_i$")
    a1.plot(centres, centres - 0.5, "--", color="0.35", label=r"Weighted Mid")
    a1.set_xlabel(r"Queue Imbalance $I = q_b/(q_b+q_a)$")
    a1.set_ylabel(r"Mid Change After $k$ Moves (ticks)")
    a1.set_ylim(-0.75, 1.0)
    a1.legend(loc="upper left", fontsize=7, ncol=2)
    panel_label(a1, "(a)")

    out = {}
    for name, pred, mk in (("Weighted Mid", wmid, "s"), ("Micro-Price", micro, "o")):
        rel, rse = [], []
        for H in HORIZONS:
            tgt, ok = future(H)
            sel = np.flatnonzero(test & ok & np.isfinite(micro))
            e_mid = (tgt[sel] - m0[sel]) ** 2
            e_x = (tgt[sel] - pred[sel]) ** 2
            blocks = np.array_split(np.arange(len(sel)), 20)
            r_b = [1 - e_x[b].mean() / e_mid[b].mean() for b in blocks]
            rel.append(1 - e_x.mean() / e_mid.mean())
            rse.append(np.std(r_b, ddof=1) / np.sqrt(20))
        out[name] = (np.array(rel), np.array(rse))
        a2.errorbar(HORIZONS, 100 * out[name][0], yerr=200 * out[name][1], fmt=mk + "-", ms=3.5,
                    label=name)
    a2.axhline(0, color="0.4", ls=":")
    a2.set_xscale("log")
    a2.set_xlabel(r"Horizon $h$ (s)")
    a2.set_ylabel(r"MSE Reduction vs Mid (%)")
    lo = 100 * min(out["Weighted Mid"][0].min(), out["Micro-Price"][0].min())
    hi = 100 * max(out["Weighted Mid"][0].max(), out["Micro-Price"][0].max())
    a2.set_ylim(min(-1.0, 1.4 * lo), 1.6 * hi)
    a2.legend(loc="upper right", fontsize=7.5)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig20_microprice", FIG)

    mw, sw = out["Weighted Mid"]
    mm_, sm = out["Micro-Price"]
    # Largest gap between model and test-half realized adjustments, in standard errors.
    # Edge buckets (I < 0.1 or I > 0.9) are rare and left out.
    inner = slice(1, N_IMB - 1)
    zmax = max(float(np.max(np.abs(real[k][0][inner] - cum[k - 1][inner]) / real[k][1][inner]))
               for k in real)
    write_numbers("e20_microprice", {
        "EtwentyNevents": f"{N_EVENTS:,}".replace(",", "{,}"),
        "EtwentyGmax": fmt(G[N_IMB - 1], 2),
        "EtwentyGmaxTwo": fmt(G[2 * N_IMB - 1], 2),
        "EtwentyMicroOne": fmt_pm(100 * mm_[0], 100 * sm[0], 1),
        "EtwentyWmidOne": fmt_pm(100 * mw[0], 100 * sw[0], 1),
        "EtwentyMicroTen": fmt_pm(100 * mm_[2], 100 * sm[2], 1),
        "EtwentyWmidTen": fmt_pm(100 * mw[2], 100 * sw[2], 1),
        "EtwentyMicroHundred": fmt_pm(100 * mm_[-1], 100 * sm[-1], 2),
        "EtwentyWmidHundred": fmt_pm(100 * mw[-1], 100 * sw[-1], 2),
        "EtwentyZmax": fmt(zmax, 1),
    })


if __name__ == "__main__":
    main()
