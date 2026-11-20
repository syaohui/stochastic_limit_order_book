"""Animated GIFs of the simulations behind the slides.

a1 order book           a Markovian (CST-type) book over one minute: the liquidity map
                        fills in from the left while the depth profile (right) changes
                        with every limit order, cancellation and market order
a2 queue race           1,000 copies of the balanced best-queue model start from 5 bids
                        and 3 asks; each walk stops when a queue empties, and the share
                        that has ended with a price rise climbs to the exact P(up)
a3 Hawkes cascade       one Hawkes path in its cluster form: immigrants arrive, each
                        event triggers children, and the intensity jumps and decays
a4 market maker         optimal GLFT quotes around a random mid price, the fills and the
                        inventory of one path, and the profit across 2,000 paths
a5 book in 3D           19 snapshots of the depth profile, 2 seconds apart, as a 3D
                        stack that turns 2.25 degrees per frame

Models and parameters follow gallery.py (g01, g02, g04, g07), which follows
e1, e2, e4 and e11. Time is in seconds, sizes in orders or shares, prices in
ticks. a4 uses the long-horizon GLFT quotes (Theorem 2 of Gueant, Lehalle and
Fernandez-Tapia 2013), which the exact quotes match far from the horizon.

Frames are drawn at 150 dpi (1080 pixels wide for the 7.2-inch panels) and
encoded with one shared 255-colour palette and no dithering, so lines and text
stay sharp. Rendering takes several minutes of CPU time, so it runs on a
compute server:

    python experiments/animations.py              all five GIFs
    python experiments/animations.py a2 a4        selected GIFs
    python experiments/animations.py --poster DIR last frame of each as PNG (for the figure audit)
    python experiments/animations.py --sample DIR 12 evenly spaced frames of each, as a small test GIF

Output: figures/animations/aN_*.gif
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib import colors
from matplotlib.collections import LineCollection
from PIL import Image

from common import FIG
from gallery import BETA_H, CST, MU_H, N_H, P_MM, Q_MM, QA0, QB0, T_H, book_map, hawkes_cascade
from stochastic_lob import best_queues as bq, market_making as mm, markov_lob as ml
from stochastic_lob.plotting import plt, save, style_3d, use_style

OUT = FIG / "animations"
RED, BLUE, ORANGE, GREEN = "#FF2C00", "#0C5DA5", "#FF9500", "#00B945"
DPI = 150
A5_DPI = 125       # the turning 3D surface changes every pixel, so it is kept smaller
HOLD_MS = 2500     # the last frame stays this long before the loop restarts
K = 20             # price levels recorded behind each best quote


def _title(fig, text):
    return fig.text(0.5, 0.975, text, ha="center", va="top", fontsize=9.5, fontweight="bold")


def _book(n_events, seed):
    return ml.simulate_markov_lob(CST, n_events, rng=seed, burn_in=20_000, record_levels=K)


# ------------------------------------------------------------------ a1
def a1():
    rec = _book(6000, 5301)
    T, STEP, FINE = 60.0, 0.25, 0.05
    grid = np.arange(0.0, T + 1e-9, FINE)
    prices, D, bb, ba = book_map(rec, grid, K)
    mid = 0.5 * (bb + ba)
    lo, hi = mid.min() - 11, mid.max() + 11
    keep = (prices >= lo - 1) & (prices <= hi + 1)
    prices, D = prices[keep], D[:, keep]
    t_ev = rec.t
    n_frames = int(round(T / STEP))

    fig = plt.figure(figsize=(7.2, 3.0))
    title = _title(fig, "")
    a = fig.add_axes([0.075, 0.15, 0.58, 0.72])
    b = fig.add_axes([0.70, 0.15, 0.23, 0.72])
    norm = colors.TwoSlopeNorm(0.0, -6.0, 6.0)
    img = np.full((len(prices), len(grid)), np.nan)
    im = a.imshow(img, origin="lower", aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest",
                  extent=[-0.5 * FINE, T + 0.5 * FINE, prices[0] - 0.5, prices[-1] + 0.5])
    lb_, = a.step([], [], where="post", color="k", lw=0.6)
    la_, = a.step([], [], where="post", color="k", lw=0.6)
    now = a.axvline(0, color="0.3", lw=0.6)
    a.set_xlim(0, T)
    a.set_ylim(lo, hi)
    a.set_xlabel(r"Time $t$ (s)")
    a.set_ylabel(r"Price $p$ (ticks)")
    a.text(0.015, 0.965, "Asks", transform=a.transAxes, va="top", fontsize=8, color=RED, fontweight="bold")
    a.text(0.015, 0.035, "Bids", transform=a.transAxes, va="bottom", fontsize=8, color=BLUE, fontweight="bold")
    bars = b.barh(prices, np.zeros(len(prices)), height=0.85, color=RED, lw=0)
    midl = b.axhline(0, color="k", lw=0.8, ls="--")
    b.set_xlim(0, 9)
    b.set_ylim(lo, hi)
    b.set_xlabel("Depth (orders)")
    b.yaxis.set_label_position("right")
    b.tick_params(labelleft=False, labelright=True)
    b.set_ylabel(r"Price $p$ (ticks)")
    info = b.text(0.95, 0.965, "", transform=b.transAxes, ha="right", va="top", fontsize=7.5,
                  bbox=dict(fc="white", ec="0.5", lw=0.5))

    def update(i):
        t = (i + 1) * STEP
        j = int(round(t / FINE)) + 1
        img[:, :j] = np.where(D[:j] == 0, np.nan, D[:j]).T
        im.set_data(np.ma.masked_invalid(img))
        lb_.set_data(grid[:j], bb[:j] + 0.5)
        la_.set_data(grid[:j], ba[:j] - 0.5)
        now.set_xdata([t, t])
        row = D[j - 1]
        for bar, v in zip(bars, row):
            bar.set_width(abs(v))
            bar.set_color(BLUE if v < 0 else RED)
        midl.set_ydata([mid[j - 1]] * 2)
        n_ev = int(np.searchsorted(t_ev, t))
        info.set_text(f"Spread {int(ba[j - 1] - bb[j - 1])} Tick" + ("s" if ba[j - 1] - bb[j - 1] > 1 else ""))
        title.set_text(f"A Markovian Limit Order Book: t = {t:4.1f} s, {n_ev:,} Events")

    return fig, update, n_frames, 20, DPI


# ------------------------------------------------------------------ a2
def a2():
    rng = np.random.default_rng(5302)
    N, M, LAM = 1000, 600, 1.0
    STEP, n_frames = 0.125, 280
    tj = np.cumsum(rng.exponential(1.0 / (4 * LAM), (N, M)), axis=1)
    mv = rng.integers(0, 4, (N, M))
    qb = QB0 + np.concatenate([np.zeros((N, 1), int), np.cumsum((mv == 0).astype(int) - (mv == 1), axis=1)], axis=1)
    qa = QA0 + np.concatenate([np.zeros((N, 1), int), np.cumsum((mv == 2).astype(int) - (mv == 3), axis=1)], axis=1)
    hit = (qb == 0) | (qa == 0)
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), M)     # jump index at which a queue empties
    t_hit = np.where(first < M + 1, np.concatenate([np.zeros((N, 1)), tj], axis=1)[np.arange(N), np.minimum(first, M)],
                     np.inf)
    t_hit[first == M] = np.inf
    up = qa[np.arange(N), np.minimum(first, M)] == 0
    jit = rng.uniform(-0.28, 0.28, (N, 2))
    exact = bq.p_up_balanced(QB0, QA0)
    times = (np.arange(n_frames) + 1) * STEP

    def pos(t):
        k = np.minimum((tj <= t).sum(axis=1), first)
        return qb[np.arange(N), k], qa[np.arange(N), k]

    hist = [pos(t) for t in np.concatenate([[0.0], times])]
    s_up = np.array([((t_hit <= t) & up).mean() for t in times])
    s_dn = np.array([((t_hit <= t) & ~up).mean() for t in times])

    fig = plt.figure(figsize=(7.2, 3.0))
    title = _title(fig, "")
    a = fig.add_axes([0.07, 0.15, 0.36, 0.72])
    b = fig.add_axes([0.56, 0.15, 0.41, 0.72])
    trails = LineCollection([], colors="0.35", linewidths=0.4, alpha=0.25)
    a.add_collection(trails)
    live = a.scatter([], [], s=4, color="k", lw=0)
    dead = a.scatter([], [], s=4, lw=0, alpha=0.8)
    a.plot([QB0], [QA0], "*", color=ORANGE, ms=11, zorder=5)
    a.set_xlim(-0.6, 14)
    a.set_ylim(-0.6, 14)
    a.set_xlabel(r"Bid Queue $q_b$ (orders)")
    a.set_ylabel(r"Ask Queue $q_a$ (orders)")
    a.text(13.6, 0.9, "Ask Empty: Up", color=RED, fontsize=7.5, ha="right")
    a.text(0.55, 13.6, "Bid Empty: Down", color=BLUE, fontsize=7.5, rotation=90, va="top")
    lu, = b.plot([], [], color=RED, lw=1.6, label="Ended Up (Ask Empty)")
    ld, = b.plot([], [], color=BLUE, lw=1.6, label="Ended Down (Bid Empty)")
    b.axhline(exact, color=RED, lw=1.0, ls="--")
    b.axhline(1 - exact, color=BLUE, lw=1.0, ls="--")
    b.plot([], [], color="k", lw=1.0, ls="--", label=f"Exact Limits {exact:.3f} and {1 - exact:.3f}")
    b.set_xlim(0, times[-1])
    b.set_ylim(0, 1.0)
    b.set_xlabel(r"Time $t$ (s)")
    b.set_ylabel(f"Share of {N:,} Walks")
    b.legend(loc="upper left", fontsize=7, ncol=1)

    def update(i):
        t = times[i]
        x, y = hist[i + 1]
        done = t_hit <= t
        segs = []
        for j in np.flatnonzero(~done):
            pts = [(hist[k][0][j] + jit[j, 0], hist[k][1][j] + jit[j, 1]) for k in range(max(0, i - 7), i + 2)]
            segs.append(pts)
        trails.set_segments(segs)
        live.set_offsets(np.c_[x[~done] + jit[~done, 0], y[~done] + jit[~done, 1]])
        xd = np.where(up[done], x[done] + jit[done, 0], 0.5 * jit[done, 0])
        yd = np.where(up[done], 0.5 * jit[done, 1], y[done] + jit[done, 1])
        dead.set_offsets(np.c_[xd, yd])
        dead.set_color([RED if u else BLUE for u in up[done]])
        lu.set_data(times[:i + 1], s_up[:i + 1])
        ld.set_data(times[:i + 1], s_dn[:i + 1])
        title.set_text(f"Race of the Best Queues: t = {t:4.1f} s, {int(done.sum()):,} of {N:,} Walks Finished")

    return fig, update, n_frames, 20, DPI


# ------------------------------------------------------------------ a3
def a3():
    t, gen, par, fam = hawkes_cascade(MU_H, N_H, BETA_H, T_H, np.random.default_rng(5104))
    grid = np.linspace(0, T_H, 4001)
    lam = MU_H + np.array([np.sum(N_H * BETA_H * np.exp(-BETA_H * (g - t[t < g]))) for g in grid])
    size = np.bincount(fam)
    cm = plt.get_cmap("tab10")
    big = np.argsort(size)[::-1]
    col = {f: cm(i % 10) for i, f in enumerate(big[size[big] > 1])}
    c_ev = np.array([col.get(f, (0.6, 0.6, 0.6, 1.0)) for f in fam])
    n_frames = 250
    times = np.linspace(0, T_H, n_frames + 1)[1:]

    fig = plt.figure(figsize=(7.2, 3.4))
    title = _title(fig, "")
    a = fig.add_axes([0.075, 0.47, 0.9, 0.42])
    b = fig.add_axes([0.075, 0.13, 0.9, 0.22])
    links = LineCollection([], linewidths=0.8, alpha=0.85)
    a.add_collection(links)
    pts = a.scatter([], [], s=9, edgecolors="k", linewidths=0.25, zorder=3)
    nowa = a.axvline(0, color="0.4", lw=0.7)
    a.set_xlim(0, T_H)
    a.set_ylim(-0.6, gen.max() + 0.9)
    a.set_xlabel(r"Time $t$ (s)", labelpad=1)
    a.set_ylabel(r"Generation $g$")
    lam_line, = b.plot([], [], color=BLUE, lw=0.7)
    fill = [None]
    b.axhline(MU_H, color=RED, lw=1.0, ls="--")
    b.text(0.33 * T_H, 0.80, r"Dashed: Baseline Rate $\mu$", transform=b.get_xaxis_transform(), color=RED,
           fontsize=7.5, ha="center")
    rug, = b.plot([], [], "|", color="k", ms=4, mew=0.5)
    nowb = b.axvline(0, color="0.4", lw=0.7)
    b.set_xlim(0, T_H)
    b.set_ylim(-0.5, 1.12 * lam.max())
    b.set_xlabel(r"Time $t$ (s)")
    b.set_ylabel(r"$\lambda_t$ (1/s)")

    def update(i):
        now = times[i]
        vis = t <= now
        segs = [[(t[par[k]], gen[par[k]]), (t[k], gen[k])] for k in np.flatnonzero(vis & (par >= 0))]
        links.set_segments(segs)
        links.set_color(c_ev[vis & (par >= 0)])
        pts.set_offsets(np.c_[t[vis], gen[vis]])
        pts.set_facecolors(c_ev[vis])
        pts.set_sizes(9 + 45 * np.exp(-(now - t[vis]) / 1.5))
        g = grid <= now
        lam_line.set_data(grid[g], lam[g])
        if fill[0] is not None:
            fill[0].remove()
        fill[0] = b.fill_between(grid[g], 0, lam[g], color=BLUE, alpha=0.25, lw=0)
        rug.set_data(t[vis], np.full(vis.sum(), -0.25))
        nowa.set_xdata([now, now])
        nowb.set_xdata([now, now])
        n_imm = int((gen[vis] == 0).sum())
        title.set_text(f"A Hawkes Cascade: t = {now:5.1f} s, {int(vis.sum())} Events "
                       f"({n_imm} Immigrants, {int(vis.sum()) - n_imm} Offspring)")

    return fig, update, n_frames, 20, DPI


# ------------------------------------------------------------------ a4
def a4():
    p = P_MM
    rng = np.random.default_rng(5304)
    db, da = mm.glft_stationary_quotes(Q_MM, p)
    N, n_steps, EVERY = 2000, 600, 2           # 120 s, one frame every 0.4 s
    S = np.zeros(N)
    q = np.zeros(N, dtype=int)
    X = np.zeros(N)
    rec = {k: np.zeros(n_steps + 1) for k in ("S", "q", "bid", "ask")}
    fills_b, fills_s = [], []
    prof = np.zeros((n_steps // EVERY + 1, N))
    rec["bid"][0], rec["ask"][0] = -db[Q_MM], da[Q_MM]
    for j in range(n_steps):
        dbj, daj = db[q + Q_MM], da[q + Q_MM]
        buy = rng.random(N) < np.clip(p.A * np.exp(-p.k * dbj) * p.dt, 0, 1)
        sell = rng.random(N) < np.clip(p.A * np.exp(-p.k * daj) * p.dt, 0, 1)
        if buy[0]:
            fills_b.append(((j + 1) * p.dt, S[0] - dbj[0]))
        if sell[0]:
            fills_s.append(((j + 1) * p.dt, S[0] + daj[0]))
        X += np.where(sell, S + daj, 0.0) - np.where(buy, S - dbj, 0.0)
        q += buy.astype(int) - sell.astype(int)
        S += p.sigma * np.sqrt(p.dt) * np.where(rng.random(N) < 0.5, 1.0, -1.0)
        rec["S"][j + 1], rec["q"][j + 1] = S[0], q[0]
        rec["bid"][j + 1], rec["ask"][j + 1] = S[0] - db[q[0] + Q_MM], S[0] + da[q[0] + Q_MM]
        if (j + 1) % EVERY == 0:
            prof[(j + 1) // EVERY] = X + q * S
    tt = np.arange(n_steps + 1) * p.dt
    fills_b, fills_s = np.array(fills_b), np.array(fills_s)
    n_frames = n_steps // EVERY
    edges = np.linspace(min(np.percentile(prof[n_frames // 10:], 0.2), 0.0) - 10,
                        np.percentile(prof[-1], 99.8) + 10, 71)

    fig = plt.figure(figsize=(7.2, 3.3))
    title = _title(fig, "")
    a = fig.add_axes([0.08, 0.45, 0.50, 0.44])
    c = fig.add_axes([0.08, 0.13, 0.50, 0.22])
    h = fig.add_axes([0.68, 0.13, 0.30, 0.76])
    lS, = a.plot([], [], color="k", lw=1.0, label="Mid")
    lB, = a.plot([], [], color=BLUE, lw=0.7, label="Bid Quote")
    lA, = a.plot([], [], color=RED, lw=0.7, label="Ask Quote")
    fb = a.scatter([], [], marker="^", s=16, color=BLUE, edgecolors="k", linewidths=0.3, zorder=4, label="We Buy")
    fs = a.scatter([], [], marker="v", s=16, color=RED, edgecolors="k", linewidths=0.3, zorder=4, label="We Sell")
    a.set_xlim(0, tt[-1])
    span = max(rec["ask"].max(), rec["S"].max()) - min(rec["bid"].min(), rec["S"].min())
    a.set_ylim(rec["bid"].min() - 0.5, rec["ask"].max() + 0.45 * span)
    a.set_xlabel(r"Time $t$ (s)", labelpad=1)
    a.set_ylabel("Price (ticks)")
    a.legend(loc="upper left", ncol=5, fontsize=6.5, handlelength=1.2, columnspacing=0.8)
    lq, = c.step([], [], where="post", color="k", lw=0.9)
    c.set_xlim(0, tt[-1])
    qa_ = np.abs(rec["q"]).max()
    c.set_ylim(-qa_ - 2, qa_ + 2)
    c.axhline(0, color="0.6", lw=0.6)
    c.set_xlabel(r"Time $t$ (s)")
    c.set_ylabel(r"$q_t$ (shares)")
    mid_e = 0.5 * (edges[1:] + edges[:-1])
    hb = h.bar(mid_e, np.zeros(len(mid_e)), width=np.diff(edges), color=ORANGE, alpha=0.85, lw=0)
    mline = h.axvline(0, color="k", lw=1.0)
    h.set_xlim(edges[0], edges[-1])
    dens_max = max(np.histogram(np.clip(prof[k], edges[0], edges[-1]), edges, density=True)[0].max()
                   for k in range(n_frames // 4, n_frames + 1))
    h.set_ylim(0, 1.35 * dens_max)
    h.set_xlabel("Profit (ticks)")
    h.set_ylabel("Density (1/tick)")
    htxt = h.text(0.04, 0.96, "", transform=h.transAxes, va="top", fontsize=7.5, bbox=dict(fc="white", ec="0.5", lw=0.5))

    def update(i):
        j = (i + 1) * EVERY
        s = slice(0, j + 1)
        lS.set_data(tt[s], rec["S"][s])
        lB.set_data(tt[s], rec["bid"][s])
        lA.set_data(tt[s], rec["ask"][s])
        now = tt[j]
        fb.set_offsets(fills_b[fills_b[:, 0] <= now] if len(fills_b) else np.empty((0, 2)))
        fs.set_offsets(fills_s[fills_s[:, 0] <= now] if len(fills_s) else np.empty((0, 2)))
        lq.set_data(tt[s], rec["q"][s])
        pr = prof[i + 1]
        dens = np.histogram(np.clip(pr, edges[0], edges[-1]), edges, density=True)[0]
        for bar, v in zip(hb, dens):
            bar.set_height(v)
        mline.set_xdata([pr.mean()] * 2)
        htxt.set_text(f"{N:,} Paths\nMean {pr.mean():.0f}\nSD {pr.std(ddof=1):.0f}")
        title.set_text(f"A Market Maker With Optimal GLFT Quotes: t = {now:5.1f} s, Inventory {int(rec['q'][j]):+d}")

    return fig, update, n_frames, 24, DPI


# ------------------------------------------------------------------ a5
def _bars(edges, h):
    v = [(edges[0], 0.0)]
    for k in range(len(h)):
        v += [(edges[k], h[k]), (edges[k + 1], h[k])]
    v.append((edges[-1], 0.0))
    return v


def a5():
    from matplotlib.collections import PolyCollection

    rec = _book(4000, 5305)
    snaps = np.arange(2.0, 38.1, 2.0)
    prices, D, bb, ba = book_map(rec, snaps, K)
    W = 12
    n = 160
    fig = plt.figure(figsize=(6.0, 4.6))
    _title(fig, f"The Book in 3D: {len(snaps)} Snapshots of the Depth, 2 s Apart")
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.93], projection="3d")
    zmax = 0.0
    for k, t0 in enumerate(snaps):
        b, a = int(bb[k]), int(ba[k])
        m = 0.5 * (b + a)
        pb = np.arange(b - W + 1, b + 1)
        pa = np.arange(a, a + W)
        hb = -D[k, pb - prices[0]]
        ha = D[k, pa - prices[0]]
        zmax = max(zmax, hb.max(), ha.max())
        polys = [_bars(np.append(pb - 0.5, b + 0.5) - m, hb), _bars(np.append(pa - 0.5, pa[-1] + 0.5) - m, ha)]
        ax.add_collection3d(PolyCollection(polys, facecolors=[BLUE, RED], edgecolors="0.15", linewidths=0.3,
                                           alpha=0.8), zs=t0, zdir="x")
    ax.set_xlim(0, snaps[-1] + 2.0)
    ax.set_ylim(-W - 1, W + 1)
    ax.set_zlim(0, zmax)
    ax.set_xlabel(r"Time $t$ (s)", labelpad=0)
    ax.set_ylabel("Distance From Mid (ticks)", labelpad=0)
    ax.set_zlabel("Depth (orders)", labelpad=-1)
    ax.set_box_aspect((1.6, 1.0, 0.7), zoom=0.85)
    ax.grid(False)
    fig.text(0.98, 0.90, "Blue: Bids   Red: Asks", ha="right", fontsize=8)

    def update(i):
        style_3d(ax, elev=26 + 8 * np.sin(2 * np.pi * i / n), azim=-60 + 360 * i / n)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.96, 0.96, 0.96, 1.0))

    return fig, update, n, 20, A5_DPI


BUILDERS = {"a1": (a1, "a1_order_book"), "a2": (a2, "a2_queue_race"), "a3": (a3, "a3_hawkes_cascade"),
            "a4": (a4, "a4_market_maker"), "a5": (a5, "a5_book_3d")}


def _grab(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def write_gif(fig, update, idx, fps, path, dpi=DPI):
    """Render frames `idx` and write one GIF with a shared palette and no dithering."""
    fig.set_dpi(dpi)
    frames = []
    for i in idx:
        update(i)
        frames.append(_grab(fig))
    pick = np.unique(np.linspace(0, len(frames) - 1, min(10, len(frames))).astype(int))
    w, h = frames[0].size
    strip = Image.new("RGB", (w, h * len(pick)), "white")
    for k, j in enumerate(pick):
        strip.paste(frames[j], (0, k * h))
    pal = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    out = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    dur = [int(round(1000 / fps))] * len(out)
    dur[-1] = HOLD_MS
    out[0].save(path, save_all=True, append_images=out[1:], duration=dur, loop=0, disposal=1)
    return w, h


def main(argv):
    use_style()
    mode, target = None, None
    for flag in ("--poster", "--sample"):
        if flag in argv:
            mode, target = flag, Path(argv[argv.index(flag) + 1])
            argv = [a for a in argv if a not in (flag, str(target))]
    names = [a for a in argv if a in BUILDERS] or list(BUILDERS)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        build, stem = BUILDERS[name]
        fig, update, n, fps, dpi = build()
        if mode == "--poster":
            update(n - 1)
            save(fig, stem + "_poster", target)
        elif mode == "--sample":
            target.mkdir(parents=True, exist_ok=True)
            idx = np.unique(np.linspace(0, n - 1, 12).astype(int))
            path = target / f"{stem}_sample.gif"
            w, h = write_gif(fig, update, idx, fps, path, dpi)
            print(f"{stem}: {n} frames at {fps} fps, {w}x{h} px; sample {path.stat().st_size / 1e3 / len(idx):.0f} kB/frame",
                  flush=True)
        else:
            path = OUT / f"{stem}.gif"
            w, h = write_gif(fig, update, range(n), fps, path, dpi)
            print(f"{stem}.gif  {n} frames  {fps} fps  {w}x{h}  {path.stat().st_size / 1e6:.1f} MB", flush=True)
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
