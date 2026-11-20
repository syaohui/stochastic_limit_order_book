"""Gallery: one showcase figure per topic that shows the randomness itself.

The experiments e1 to e23 measure models and estimators. These figures show the
simulated objects behind them: the book as it evolves, queue races, branching
cascades, latent liquidity, quote surfaces and sample paths. Models and
parameters follow the matching experiments unless a line below says otherwise.

g01 Markovian book      liquidity map of one simulated book; depth profiles at six times in 3D
g02 best queues         race of the two best queues to zero; P(up) surface of Cont and de Larrard
g03 queue-reactive      queue-size paths from a long queue; its law relaxing to the stationary one
g04 Hawkes              branching cascade of one Hawkes path and its intensity
g05 price impact        latent order book during and after a metaorder; noisy metaorder impact paths
g06 order flow          mid, weighted mid and micro-price over 15 seconds; law of the next
                        two-second mid change by queue imbalance
g07 market making       exact GLFT bid surface; inventory paths coloured by final profit
g08 adverse selection   Glosten-Milgrom belief paths; spread surface over belief and informed share
g09 AMM                 cumulative LVR paths; LVR per unit capital over volatility and range width

Units: time in seconds (g09: days), sizes in orders, prices in ticks, unless a
panel says otherwise. The simulations take about half a minute; the cache
results/raw/gallery.npz lets the plots be redrawn alone (--plot).

Output: figures/gallery/gNN_*.png, results/gallery.{json,tex}
"""
import numpy as np
from matplotlib import colormaps, colors
from matplotlib.collections import LineCollection, PolyCollection
from scipy.linalg import expm
from scipy.special import roots_legendre

from common import FIG, fmt, fmt_int, fmt_pm, load_raw, run, save_raw, write_numbers
from stochastic_lob import adverse_selection as gm
from stochastic_lob import best_queues as bq
from stochastic_lob import impact, latent_book as lb, market_making as mm
from stochastic_lob import markov_lob as ml, microprice as mp, queue_reactive as qr
from stochastic_lob.plotting import panel_label, plt, save, style_3d, use_style

NAME = "gallery"
OUT = FIG / "gallery"
TALL = (7.2, 3.1)
RED, BLUE, ORANGE, GREEN, GREY = "#FF2C00", "#0C5DA5", "#FF9500", "#00B945", "0.55"

# g01, g06: the CST-type book of e1, e8 and e20
CST = ml.power_law_params(L=30, lam1=1.9, a=0.5, theta0=0.5, mu=1.0)
K_LEVELS = 25            # price levels recorded behind each best quote
N_IMB = 10               # imbalance buckets of the micro-price (as e20)
H_MOVE = 2.0             # horizon of the mid change in g06 (b) (s)
W_MICRO = 15.0           # length of the stretch shown in g06 (a) (s)
# g02: balanced best queues (limit rate = market + cancel rate), start q_b = 5, q_a = 3
QB0, QA0 = 5, 3
WIN2 = 20                # panel (a) shows queues up to this size (a few long excursions leave it)
# g03: the illustrative queue-reactive rates of e3, started from a long queue
Q0_QR, QMAX_QR = 15, 60
# g04: exponential Hawkes kernel n beta exp(-beta t)
MU_H, N_H, BETA_H, T_H = 0.15, 0.85, 0.8, 200.0
# g05: latent book with L = D = 1, metaorder rate m = 20 shares/s for T = 1 s (u = 20)
M_LB, T_LB = 20.0, 1.0
GAMMA_EPS, ZETA = 0.5, 0.25        # sign memory exponent and the critical propagator exponent
N_TR, T0_MO, N_MO, RHO_MO = 3000, 500, 1000, 0.3
# g07: the exact GLFT problem of e11
P_MM = mm.MMParams(S0=0.0, T=600.0, sigma=0.3, dt=0.2, gamma=0.01, k=0.3, A=0.9)
Q_MM = 30
# g08: Glosten-Milgrom with 30% informed traders, values 0 and 1
ALPHA_GM = 0.3
# g09: daily volatility 4% (as e13), 30 days
SIGMA_AMM, T_AMM = 0.04, 30.0


# ------------------------------------------------------------------ simulations
def book_map(rec, t_grid, K):
    """Signed depth at absolute price levels (asks > 0, bids < 0), sampled on t_grid."""
    k = np.clip(np.searchsorted(rec.t, t_grid, side="right") - 1, 0, None)
    bb, ba = rec.best_bid[k], rec.best_ask[k]
    if np.abs(ba - bb).max() > 10 ** 6:
        raise RuntimeError("one side of the book was empty at a sample time")
    lo, hi = int(bb.min()) - K + 1, int(ba.max()) + K - 1
    prices = np.arange(lo, hi + 1)
    D = np.zeros((len(t_grid), len(prices)), dtype=np.int32)
    m = np.arange(K)
    rows = np.arange(len(t_grid))[:, None]
    D[rows, bb[:, None] - m[None, :] - lo] = -rec.bid_depth[k]
    D[rows, ba[:, None] + m[None, :] - lo] = rec.ask_depth[k]
    return prices, D, bb, ba


def queue_race(qb0, qa0, n, rng, n_keep=150, chunk=400, max_chunks=400):
    """Jump chain of the balanced best-queue model until one queue is empty.

    Each event moves one queue up or down by one order, the four moves with
    equal probability. Returns up (the ask queue emptied first), the number
    of unfinished paths, and the lattice paths of the first n_keep walks.
    """
    qb = np.full(n, qb0)
    qa = np.full(n, qa0)
    alive = np.ones(n, bool)
    up = np.zeros(n, bool)
    keep = [[(qb0, qa0)] for _ in range(n_keep)]
    for _ in range(max_chunks):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        mv = rng.integers(0, 4, (idx.size, chunk))
        b = qb[idx, None] + np.cumsum((mv == 0).astype(int) - (mv == 1), axis=1)
        a = qa[idx, None] + np.cumsum((mv == 2).astype(int) - (mv == 3), axis=1)
        hit = (b == 0) | (a == 0)
        done = hit.any(axis=1)
        first = np.where(done, hit.argmax(axis=1), chunk - 1)
        for j in np.flatnonzero(idx < n_keep):
            keep[idx[j]].extend(zip(b[j, :first[j] + 1], a[j, :first[j] + 1]))
        r = np.arange(idx.size)
        up[idx[done]] = a[r[done], first[done]] == 0
        qb[idx] = b[r, first]
        qa[idx] = a[r, first]
        alive[idx[done]] = False
    return up, int(alive.sum()), [np.array(p) for p in keep]


def qr_generator(rates, q_max):
    q = np.arange(q_max + 1)
    up = np.where(q < q_max, rates.up(q), 0.0)
    dn = rates.down(q)
    G = np.diag(up[:-1], 1) + np.diag(dn[1:], -1)
    return G - np.diag(G.sum(axis=1))


def qr_paths(rates, q0, t_end, n, rng):
    out = []
    for _ in range(n):
        t, q, ts, qs = 0.0, q0, [0.0], [q0]
        while True:
            r_up, r_dn = float(rates.up(q)), float(rates.down(q))
            t += rng.exponential(1.0 / (r_up + r_dn))
            if t > t_end:
                break
            q += 1 if rng.random() * (r_up + r_dn) < r_up else -1
            ts.append(t)
            qs.append(q)
        ts.append(t_end)
        qs.append(q)
        out.append(np.array([ts, qs], float))
    return out


def hawkes_cascade(mu, n, beta, T, rng):
    """Cluster form of an exponential Hawkes process on [0, T].

    Immigrants arrive as a Poisson process of rate mu (1/s). Each event has a
    Poisson(n) number of children, each after an exponential delay of mean
    1/beta (s). Returns times, generation, parent index (-1 for immigrants) and
    family (index of the immigrant), sorted by time.
    """
    t0 = np.sort(rng.uniform(0, T, rng.poisson(mu * T)))
    times, gen, par, fam = list(t0), [0] * len(t0), [-1] * len(t0), list(range(len(t0)))
    k = 0
    while k < len(times):
        for _ in range(rng.poisson(n)):
            tc = times[k] + rng.exponential(1.0 / beta)
            if tc < T:
                times.append(tc)
                gen.append(gen[k] + 1)
                par.append(k)
                fam.append(fam[k])
        k += 1
    times = np.array(times)
    order = np.argsort(times, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    par = np.array(par)
    par = np.where(par >= 0, rank[np.clip(par, 0, None)], -1)
    return times[order], np.array(gen)[order], par[order], np.array(fam)[order]


def latent_phi(x, t, m, T, L=1.0, D=1.0, n_nodes=400):
    """Latent book phi(x, t) (shares/tick) for a buy metaorder at rate m on [0, T].

    phi = -L x + m int_0^min(t,T) G(x - y_s, t - s) ds with the heat kernel G and
    the self-similar price y_s = sqrt(D s) F(m/(L D)). With s = t - w^2,
    phi = -L x + (m / sqrt(pi D)) int exp(-(x - y_{t - w^2})^2 / (4 D w^2)) dw,
    over sqrt(max(t - T, 0)) <= w <= sqrt(t), by Gauss-Legendre quadrature.
    """
    A = lb.self_similar_amplitude(m / (L * D)) * np.sqrt(D)
    z, wq = roots_legendre(n_nodes)
    out = np.empty((len(t), len(x)))
    for j, tj in enumerate(t):
        a, b = np.sqrt(max(tj - T, 0.0)), np.sqrt(tj)
        w = 0.5 * (b - a) * z + 0.5 * (b + a)
        y = A * np.sqrt(np.clip(tj - w * w, 0, None))
        g = np.exp(-(x[None, :] - y[:, None]) ** 2 / (4 * D * w[:, None] ** 2))
        out[j] = -L * x + m / np.sqrt(np.pi * D) * 0.5 * (b - a) * (wq @ g)
    return out, A


def zero_crossing(x, phi_row):
    s = np.flatnonzero((phi_row[:-1] > 0) & (phi_row[1:] <= 0))
    j = s[0]
    return x[j] + phi_row[j] / (phi_row[j] - phi_row[j + 1]) * (x[j + 1] - x[j])


def glft_paths(p, Q, n_paths, rng):
    """Market maker with exact GLFT quotes; same dynamics as market_making.simulate."""
    n_steps = int(round(p.T / p.dt))
    db, da = mm.glft_exact_quotes(Q, p, p.T - np.arange(n_steps) * p.dt)
    S = np.full(n_paths, p.S0)
    q = np.zeros(n_paths, dtype=int)
    X = np.zeros(n_paths)
    qp = np.empty((n_steps + 1, n_paths), dtype=np.int16)
    qp[0] = 0
    for j in range(n_steps):
        dbj, daj = db[j, q + Q], da[j, q + Q]
        buy = rng.random(n_paths) < np.clip(p.A * np.exp(-p.k * dbj) * p.dt, 0, 1)
        sell = rng.random(n_paths) < np.clip(p.A * np.exp(-p.k * daj) * p.dt, 0, 1)
        X += np.where(sell, S + daj, 0.0) - np.where(buy, S - dbj, 0.0)
        q += buy.astype(int) - sell.astype(int)
        S += p.sigma * np.sqrt(p.dt) * np.where(rng.random(n_paths) < 0.5, 1.0, -1.0)
        qp[j + 1] = q
    return qp, X + q * S


def compute():
    rng = np.random.default_rng(5100)
    out = {}

    # g01 and g06 share one long run of the CST book
    rec = ml.simulate_markov_lob(CST, 300_000, rng=5101, burn_in=20_000, record_levels=K_LEVELS)
    t1 = np.arange(0.0, 300.0, 0.25)
    prices, D, bb, ba = book_map(rec, t1, K_LEVELS)
    snaps = np.linspace(20.0, 280.0, 6)
    sp, sD, sbb, sba = book_map(rec, snaps, K_LEVELS)
    n_win = int(np.searchsorted(rec.t, 300.0))
    out.update(g1_t=t1, g1_prices=prices, g1_D=D, g1_bb=bb, g1_ba=ba, g1_snaps=snaps,
               g1_sp=sp, g1_sD=sD, g1_sbb=sbb, g1_sba=sba, g1_events=n_win,
               g1_spread1=rec.spread_hist.get(1, 0.0))

    mid = 0.5 * (rec.best_bid + rec.best_ask).astype(float)
    spr = (rec.best_ask - rec.best_bid).astype(float)
    idx, imb = mp.states(rec.q_bid, rec.q_ask, spr, N_IMB, 2)
    h = len(mid) // 2
    G = mp.fit(idx[:h], mid[:h], N_IMB, 2)["G"]
    _, wmid, micro = mp.predictors(idx, imb, mid, spr, G)
    t_start = rec.t[h] + 100.0
    sel = (rec.t >= t_start) & (rec.t <= t_start + W_MICRO)
    out.update(g6_t=rec.t[sel] - t_start, g6_mid=mid[sel], g6_w=wmid[sel], g6_micro=micro[sel],
               g6_G=G)
    ts = np.arange(rec.t[0], rec.t[-1] - H_MOVE, 0.5)
    k0 = np.searchsorted(rec.t, ts, side="right") - 1
    k1 = np.searchsorted(rec.t, ts + H_MOVE, side="right") - 1
    ok = (spr[k0] == 1) & (rec.q_bid[k0] > 0) & (rec.q_ask[k0] > 0)
    I0 = rec.q_bid[k0][ok] / (rec.q_bid[k0][ok] + rec.q_ask[k0][ok])
    dm = (mid[k1] - mid[k0])[ok]
    bins = np.linspace(0, 1, 6)
    grid = np.arange(-2.0, 2.01, 0.5)
    pmf = np.zeros((5, len(grid)))
    nb = np.zeros(5, int)
    mean_dm = np.zeros((5, 2))
    for j in range(5):
        s = (I0 >= bins[j]) & (I0 < bins[j + 1]) if j < 4 else (I0 >= bins[j])
        nb[j] = s.sum()
        pmf[j] = [np.mean(np.isclose(dm[s], g)) for g in grid]
        mean_dm[j] = dm[s].mean(), dm[s].std(ddof=1) / np.sqrt(s.sum())
    out.update(g6_grid=grid, g6_pmf=pmf, g6_nb=nb, g6_mean=mean_dm, g6_tail=np.mean(np.abs(dm) > 2))

    # g02: queue race
    up, unfinished, paths = queue_race(QB0, QA0, 20_000, rng)
    out.update(g2_up=up.mean(), g2_se=up.std(ddof=1) / np.sqrt(up.size), g2_n=up.size,
               g2_unfinished=unfinished, g2_exact=bq.p_up_balanced(QB0, QA0))
    lens = np.array([len(p) for p in paths])
    out.update(g2_lens=lens, g2_paths=np.concatenate(paths),
               g2_pathup=np.array([p[-1, 1] == 0 for p in paths]))
    qg = np.arange(1, 13)
    out.update(g2_q=qg, g2_surf=np.array([[bq.p_up_balanced(b, a) for a in qg] for b in qg]))

    # g03: queue-reactive transient law
    rates = qr.illustrative_rates()
    Gq = qr_generator(rates, QMAX_QR)
    p0 = np.zeros(QMAX_QR + 1)
    p0[Q0_QR] = 1.0
    t_f = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    law = np.array([p0 @ expm(Gq * t) for t in t_f])
    pi = qr.stationary_distribution(rates, QMAX_QR)
    t_m = np.linspace(0, 60, 241)
    q = np.arange(QMAX_QR + 1)
    mean_q = np.array([(p0 @ expm(Gq * t)) @ q for t in t_m])
    qp = qr_paths(rates, Q0_QR, 60.0, 80, rng)
    out.update(g3_tf=t_f, g3_law=law, g3_pi=pi, g3_tm=t_m, g3_mean=mean_q,
               g3_lens=np.array([p.shape[1] for p in qp]), g3_paths=np.concatenate(qp, axis=1),
               g3_tv32=0.5 * np.abs(law[-1] - pi).sum())

    # g04: Hawkes cascade
    ht, hg, hp, hf = hawkes_cascade(MU_H, N_H, BETA_H, T_H, np.random.default_rng(5104))
    grid = np.linspace(0, T_H, 4001)
    lam = MU_H + np.array([np.sum(N_H * BETA_H * np.exp(-BETA_H * (g - ht[ht < g]))) for g in grid])
    out.update(g4_t=ht, g4_gen=hg, g4_par=hp, g4_fam=hf, g4_grid=grid, g4_lam=lam)

    # g05: latent book surface and metaorder paths in the propagator model
    xf = np.linspace(-5.0, 13.0, 3601)
    tf = np.linspace(0.02, 3.0, 150)
    phif, A = latent_phi(xf, tf, M_LB, T_LB)
    yf = np.array([zero_crossing(xf, r) for r in phif])
    during = tf <= T_LB
    err = np.max(np.abs(yf[during] / (A * np.sqrt(tf[during])) - 1))
    t_fen = np.array([0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    x_fen = np.linspace(-4.0, 10.0, 561)
    phi_fen, _ = latent_phi(x_fen, t_fen, M_LB, T_LB)
    y_fen = np.array([zero_crossing(x_fen, r) for r in phi_fen])
    out.update(g5_tf=tf, g5_y=yf, g5_A=A, g5_err=err, g5_tfen=t_fen, g5_xfen=x_fen,
               g5_phifen=phi_fen, g5_yfen=y_fen)
    mo = np.zeros(N_TR)
    mo[T0_MO:T0_MO + N_MO] = RHO_MO
    theory = impact.propagator_price(mo, 1.0, ZETA)
    P = np.empty((200, N_TR))
    for r in range(200):
        x = impact.long_memory_signs(N_TR, GAMMA_EPS, rng)
        ours = np.zeros(N_TR, bool)
        ours[T0_MO:T0_MO + N_MO] = rng.random(N_MO) < RHO_MO
        x[ours] = 1.0
        p = impact.propagator_price(x, 1.0, ZETA)
        P[r] = p - p[T0_MO]
    out.update(g5_P=P.astype(np.float32), g5_theory=theory - theory[T0_MO])

    # g07: exact GLFT quotes and inventory paths
    tt = np.linspace(0, P_MM.T, 61)
    db, _ = mm.glft_exact_quotes(Q_MM, P_MM, P_MM.T - tt)
    qpath, profit = glft_paths(P_MM, Q_MM, 300, rng)
    out.update(g7_t=tt, g7_db=db[:, :-1], g7_q=qpath[::5].astype(np.int16), g7_profit=profit)

    # g08: Glosten-Milgrom
    sim = gm.simulate(ALPHA_GM, 60, 2000, rng=rng)
    pis = np.linspace(0.005, 0.995, 60)
    als = np.linspace(0.0, 0.9, 46)
    bid, ask, _, _ = gm.quotes(pis[:, None], als[None, :])
    out.update(g8_belief=sim["belief"][:200], g8_V=sim["V"][:200], g8_spread=sim["spread"].mean(axis=0),
               g8_pis=pis, g8_als=als, g8_surf=ask - bid,
               g8_wrong=np.mean(np.abs(sim["belief"][:, -1] - sim["V"]) > 0.5))

    # g09: cumulative LVR paths of a constant-product pool, every-step arbitrage
    n_steps, n_paths = int(T_AMM * 96), 400
    dt = T_AMM / n_steps
    logP = np.zeros(n_paths)
    R = np.full(n_paths, 2.0)
    lvr = np.zeros((n_steps + 1, n_paths))
    for j in range(n_steps):
        P_old = np.exp(logP)
        logP += -0.5 * SIGMA_AMM ** 2 * dt + SIGMA_AMM * np.sqrt(dt) * rng.standard_normal(n_paths)
        R += (np.exp(logP) - P_old) / np.sqrt(P_old)
        lvr[j + 1] = R - 2.0 * np.exp(0.5 * logP)
    out.update(g9_t=np.linspace(0, T_AMM, n_steps + 1)[::4], g9_lvr=(50.0 * lvr[::4]).astype(np.float32))

    save_raw(NAME, **out)


# ------------------------------------------------------------------ plotting helpers
def _bars(edges, h, base=0.0):
    """Outline of bars of height h between consecutive edges, as polygon vertices."""
    v = [(edges[0], base)]
    for k in range(len(h)):
        v += [(edges[k], h[k]), (edges[k + 1], h[k])]
    v.append((edges[-1], base))
    return v


def _fences(ax, positions, grid, dens, cut, body_colors, tail_color=RED, alpha=0.88, base=0.0):
    """Curves standing at `positions` along the x axis (grid along y, value along z).

    The part of slice k at or beyond cut[k] is filled with tail_color; the two
    parts do not overlap, so the 3D depth sorting cannot hide one behind the other.
    """
    for k, x0 in enumerate(positions):
        y, z = grid, np.maximum(dens[k], base)
        c = float(np.clip(cut[k], y[0], y[-1]))
        zc = float(np.interp(c, y, z))
        lo, hi = y < c, y > c
        core = [(y[0], base), *zip(y[lo], z[lo]), (c, zc), (c, base)]
        tail = [(c, base), (c, zc), *zip(y[hi], z[hi]), (y[-1], base)]
        pc = PolyCollection([core, tail], facecolors=[body_colors[k], tail_color], edgecolors="0.15",
                            linewidths=0.35, alpha=alpha)
        ax.add_collection3d(pc, zs=x0, zdir="x")


def _tag3d(ax, text, x=0.02, y=0.95):
    ax.text2D(x, y, text, transform=ax.transAxes, fontweight="bold", fontsize=9)


def _note3d(ax, text, x=0.98, y=0.88):
    ax.text2D(x, y, text, transform=ax.transAxes, ha="right", fontsize=7.5)


def _surface(ax, X, Y, Z, cmap="viridis", norm=None):
    norm = norm or colors.Normalize(np.nanmin(Z), np.nanmax(Z))
    return ax.plot_surface(X, Y, Z, facecolors=colormaps[cmap](norm(Z)), rstride=1, cstride=1,
                           linewidth=0.15, edgecolor=(0, 0, 0, 0.12), antialiased=True, shade=False)


def _cbar(fig, mappable, rect, label):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label)
    cb.ax.tick_params(direction="in", labelsize=8)
    return cb


# ------------------------------------------------------------------ figures
def g01(d):
    fig = plt.figure(figsize=TALL)
    a1 = fig.add_axes([0.07, 0.17, 0.33, 0.72])
    t, prices, D = d["g1_t"], d["g1_prices"], d["g1_D"]
    vmax = 6.0
    im = a1.pcolormesh(t, prices, D.T, cmap="RdBu_r", norm=colors.TwoSlopeNorm(0.0, -vmax, vmax),
                       shading="nearest", rasterized=True)
    a1.step(t, d["g1_bb"], where="post", color="k", lw=0.6)
    a1.step(t, d["g1_ba"], where="post", color="k", lw=0.6)
    mid = 0.5 * (d["g1_bb"] + d["g1_ba"])
    a1.set_ylim(mid.min() - 9, mid.max() + 9)
    a1.set_xlim(0, t[-1])
    a1.set_xlabel(r"Time $t$ (s)")
    a1.set_ylabel(r"Price $p$ (ticks)")
    a1.text(0.03, 0.96, "Asks", transform=a1.transAxes, va="top", fontsize=8, color=RED, fontweight="bold")
    a1.text(0.03, 0.04, "Bids", transform=a1.transAxes, va="bottom", fontsize=8, color=BLUE, fontweight="bold")
    panel_label(a1, "(a)")
    cb = _cbar(fig, im, [0.41, 0.17, 0.012, 0.72], "Signed Depth (orders)")
    cb.set_ticks([-6, -3, 0, 3, 6])

    a2 = fig.add_axes([0.47, 0.0, 0.53, 1.0], projection="3d")
    sp, sD = d["g1_sp"], d["g1_sD"]
    W = 12                                           # levels shown on each side of the mid
    zmax = 0.0
    for k, t0 in enumerate(d["g1_snaps"]):
        b, a = int(d["g1_sbb"][k]), int(d["g1_sba"][k])
        m = 0.5 * (b + a)
        pb = np.arange(b - W + 1, b + 1)            # bid levels, outermost first
        pa = np.arange(a, a + W)
        hb = -sD[k, pb - sp[0]]
        ha = sD[k, pa - sp[0]]
        zmax = max(zmax, hb.max(), ha.max())
        bid = _bars(np.append(pb - 0.5, b + 0.5) - m, hb)
        ask = _bars(np.append(pa - 0.5, pa[-1] + 0.5) - m, ha)
        pc = PolyCollection([bid, ask], facecolors=[BLUE, RED], edgecolors="0.15", linewidths=0.35, alpha=0.88)
        a2.add_collection3d(pc, zs=t0, zdir="x")
    style_3d(a2, elev=26, azim=-60)
    a2.set_xlim(0, 300)
    a2.set_ylim(-W - 1, W + 1)
    a2.set_zlim(0, zmax)
    a2.set_box_aspect(None, zoom=0.78)
    a2.set_xlabel(r"Time $t$ (s)", labelpad=0)
    a2.set_ylabel(r"Distance From Mid (ticks)", labelpad=0)
    a2.set_zlabel("Depth (orders)", labelpad=-1)
    _note3d(a2, "Blue: Bids   Red: Asks")
    _tag3d(a2, "(b)")
    save(fig, "g01_markov_book", OUT)
    return {"GalOneEvents": fmt_int(d["g1_events"]), "GalOneSpreadOne": f"{100 * float(d['g1_spread1']):.0f}"}


def g02(d):
    fig = plt.figure(figsize=TALL)
    a1 = fig.add_axes([0.075, 0.17, 0.36, 0.66])
    lens, allp, pu = d["g2_lens"], d["g2_paths"], d["g2_pathup"]
    starts = np.concatenate([[0], np.cumsum(lens)])
    rng = np.random.default_rng(7)
    for j in range(len(lens)):
        p = allp[starts[j]:starts[j + 1]].astype(float)
        p[(p > WIN2 - 1).any(axis=1)] = np.nan        # parts of long excursions outside the frame
        off = rng.uniform(-0.22, 0.22, 2)
        a1.plot(p[:, 0] + off[0], p[:, 1] + off[1], color=RED if pu[j] else BLUE, lw=0.5, alpha=0.35)
    a1.plot([QB0], [QA0], "*", color="k", ms=10, zorder=5)
    a1.set_xlim(-0.5, WIN2)
    a1.set_ylim(-0.5, WIN2)
    a1.set_xlabel(r"Bid Queue $q_b$ (orders)")
    a1.set_ylabel(r"Ask Queue $q_a$ (orders)")
    a1.plot([], [], color=RED, label=f"Ask Empties: Price Up ({100 * float(d['g2_up']):.1f}%)")
    a1.plot([], [], color=BLUE, label="Bid Empties: Price Down")
    a1.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), fontsize=7, borderaxespad=0.0)
    panel_label(a1, "(a)")

    a2 = fig.add_axes([0.47, 0.0, 0.53, 1.0], projection="3d")
    B, A = np.meshgrid(d["g2_q"], d["g2_q"], indexing="ij")
    Z = d["g2_surf"]
    a2.computed_zorder = False
    srf = _surface(a2, B, A, Z, cmap="RdBu_r", norm=colors.Normalize(0, 1))
    srf.set_zorder(1)
    cs = a2.contour(B, A, Z, levels=[0.25, 0.5, 0.75], colors="k", linewidths=1.0)
    cs.set_zorder(5)
    ex = float(d["g2_exact"])
    a2.scatter([QB0], [QA0], [ex], color="k", s=30, marker="*", zorder=6, depthshade=False)
    style_3d(a2, elev=26, azim=-128)
    a2.set_box_aspect(None, zoom=0.82)
    a2.set_zlim(0, 1)
    a2.set_xlabel(r"Bid Queue $q_b$ (orders)", labelpad=0)
    a2.set_ylabel(r"Ask Queue $q_a$ (orders)", labelpad=0)
    a2.set_zlabel("P(Next Move Up)", labelpad=-1)
    _note3d(a2, "Black: 25%, 50%, 75% Levels")
    _tag3d(a2, "(b)")
    save(fig, "g02_best_queues", OUT)
    return {"GalTwoExact": fmt(ex, 3), "GalTwoMC": fmt_pm(float(d["g2_up"]), float(d["g2_se"]), 3),
            "GalTwoPaths": fmt_int(d["g2_n"]), "GalTwoZ": fmt(abs(float(d["g2_up"]) - ex) / float(d["g2_se"]), 1)}


def g03(d):
    fig = plt.figure(figsize=TALL)
    a1 = fig.add_axes([0.075, 0.17, 0.38, 0.72])
    lens, allp = d["g3_lens"], d["g3_paths"]
    starts = np.concatenate([[0], np.cumsum(lens)])
    for j in range(len(lens)):
        p = allp[:, starts[j]:starts[j + 1]]
        a1.step(p[0], p[1], where="post", color=BLUE, lw=0.5, alpha=0.18)
    a1.step(p[0], p[1], where="post", color="0.15", lw=0.8, label="One Path")
    a1.plot(d["g3_tm"], d["g3_mean"], color=RED, lw=1.6, label=r"Mean $E[q_t]$")
    m_pi = float(d["g3_pi"] @ np.arange(len(d["g3_pi"])))
    a1.axhline(m_pi, color="k", lw=0.9, ls="--", label="Stationary Mean")
    a1.set_xlim(0, 60)
    a1.set_ylim(-0.5, 22)
    a1.set_xlabel(r"Time $t$ (s)")
    a1.set_ylabel(r"Queue Size $q_t$ (orders)")
    a1.legend(loc="upper right", fontsize=7)
    panel_label(a1, "(a)")

    a2 = fig.add_axes([0.47, 0.0, 0.53, 1.0], projection="3d")
    qmax = 22
    edges = np.arange(-0.5, qmax + 1.0)
    cm = colormaps["viridis"]
    tf, law = d["g3_tf"][1:], d["g3_law"][1:]       # t = 0 is the point mass at q0
    pos = np.arange(len(tf) + 1)
    for k, t0 in enumerate(tf):
        pc = PolyCollection([_bars(edges, law[k, :qmax + 1])], facecolors=[cm(k / len(pos))],
                            edgecolors="0.15", linewidths=0.3, alpha=0.85)
        a2.add_collection3d(pc, zs=pos[k], zdir="x")
    pc = PolyCollection([_bars(edges, d["g3_pi"][:qmax + 1])], facecolors=[RED], edgecolors="0.15",
                        linewidths=0.3, alpha=0.85)
    a2.add_collection3d(pc, zs=pos[-1], zdir="x")
    style_3d(a2, elev=24, azim=-62)
    a2.set_xlim(0, pos[-1])
    a2.set_xticks(pos)
    a2.set_xticklabels([f"{t:g}" for t in tf] + [r"$\infty$"])
    a2.set_ylim(-0.5, qmax + 0.5)
    a2.set_zlim(0, 0.3)
    a2.set_box_aspect(None, zoom=0.82)
    a2.set_xlabel(r"Time $t$ (s)", labelpad=0)
    a2.set_ylabel(r"Queue Size $q$ (orders)", labelpad=0)
    a2.set_zlabel("Probability", labelpad=-1)
    _note3d(a2, f"Start: $q_0$ = {Q0_QR}   " + r"Red: Stationary Law")
    _tag3d(a2, "(b)")
    save(fig, "g03_queue_reactive", OUT)
    pi = d["g3_pi"]
    return {"GalThreeMean": fmt(float(pi @ np.arange(len(pi))), 2), "GalThreeTV": fmt(float(d["g3_tv32"]), 3)}


def g04(d):
    fig = plt.figure(figsize=(7.2, 3.8))
    a1 = fig.add_axes([0.075, 0.47, 0.9, 0.46])
    a2 = fig.add_axes([0.075, 0.11, 0.9, 0.23])
    t, gen, par, fam = d["g4_t"], d["g4_gen"], d["g4_par"], d["g4_fam"]
    size = np.bincount(fam)
    cm = colormaps["tab10"]
    big = np.argsort(size)[::-1]
    col = {f: cm(i % 10) for i, f in enumerate(big[size[big] > 1])}
    c_ev = [col.get(f, (0.6, 0.6, 0.6, 1.0)) for f in fam]
    segs = [[(t[par[i]], gen[par[i]]), (t[i], gen[i])] for i in range(len(t)) if par[i] >= 0]
    sc = [c_ev[i] for i in range(len(t)) if par[i] >= 0]
    a1.add_collection(LineCollection(segs, colors=sc, linewidths=0.8, alpha=0.85))
    a1.scatter(t, gen, c=c_ev, s=9, edgecolors="k", linewidths=0.25, zorder=3)
    a1.set_ylim(-0.6, gen.max() + 0.9)
    a1.set_ylabel(r"Generation $g$")
    a1.set_xlim(0, T_H)
    a1.set_xlabel(r"Time $t$ (s)")
    a1.text(0.995, 0.93, f"{len(t)} Events: {int((gen == 0).sum())} Immigrants, "
            f"{int((gen > 0).sum())} Offspring", transform=a1.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(fc="white", ec="0.5", lw=0.5))
    panel_label(a1, "(a)")
    a2.fill_between(d["g4_grid"], 0, d["g4_lam"], color=BLUE, alpha=0.25, lw=0)
    a2.plot(d["g4_grid"], d["g4_lam"], color=BLUE, lw=0.7)
    a2.axhline(MU_H, color=RED, lw=1.0, ls="--")
    a2.text(0.33 * T_H, 0.80, r"Dashed: Baseline Rate $\mu$", transform=a2.get_xaxis_transform(),
            color=RED, fontsize=7.5, ha="center")
    a2.plot(t, np.full(len(t), -0.25), "|", color="k", ms=4, mew=0.5)
    a2.set_xlim(0, T_H)
    a2.set_ylim(-0.5, 1.12 * d["g4_lam"].max())
    a2.set_xlabel(r"Time $t$ (s)")
    a2.set_ylabel(r"$\lambda_t$ (1/s)")
    panel_label(a2, "(b)")
    save(fig, "g04_hawkes", OUT)
    return {"GalFourEvents": str(len(t)), "GalFourExpected": f"{MU_H * T_H / (1 - N_H):.0f}",
            "GalFourImm": str(int((gen == 0).sum())), "GalFourLargest": str(int(size.max())),
            "GalFourMaxGen": str(int(gen.max())), "GalFourPeak": fmt(float(d["g4_lam"].max()), 1)}


def g05(d):
    fig = plt.figure(figsize=TALL)
    a1 = fig.add_axes([0.47, 0.0, 0.53, 1.0], projection="3d")
    tfen, xfen = d["g5_tfen"], d["g5_xfen"]
    _fences(a1, tfen, xfen, np.abs(d["g5_phifen"]), d["g5_yfen"], [BLUE] * len(tfen), tail_color=RED)
    a1.plot(d["g5_tf"], d["g5_y"], np.zeros_like(d["g5_y"]), color="k", lw=1.6)
    a1.scatter(tfen, d["g5_yfen"], np.zeros_like(tfen), color="k", s=10, depthshade=False)
    style_3d(a1, elev=26, azim=-52)
    a1.set_xlim(0, 3.0)
    a1.set_ylim(xfen[0], xfen[-1])
    a1.set_zlim(0, 10)
    a1.set_box_aspect(None, zoom=0.78)
    a1.set_xlabel(r"Time $t$ (s)", labelpad=0)
    a1.set_ylabel(r"Price $x$ (ticks)", labelpad=0)
    a1.set_zlabel("Latent Volume (shares/tick)", labelpad=-1)
    _note3d(a1, "Blue: Bids   Red: Asks   Black: Price")
    _tag3d(a1, "(b)")

    a2 = fig.add_axes([0.075, 0.17, 0.38, 0.72])
    P = d["g5_P"]
    k = np.arange(N_TR) - T0_MO
    for r in range(100):
        a2.plot(k, P[r], color=BLUE, lw=0.4, alpha=0.15)
    a2.plot(k, P.mean(axis=0), color=RED, lw=1.6, label=f"Mean of {len(P)} Runs")
    a2.plot(k, d["g5_theory"], color="k", lw=1.0, ls="--", label="Propagator Theory")
    a2.axvspan(0, N_MO, color=ORANGE, alpha=0.12, lw=0)
    a2.text(N_MO / 2, 0.05, "Metaorder", transform=a2.get_xaxis_transform(), ha="center", fontsize=7.5)
    lim = np.percentile(np.abs(P), 99.5)
    a2.set_ylim(-0.75 * lim, 1.6 * lim)
    a2.set_xlim(k[0], k[-1])
    a2.set_xlabel(r"Trades Since Start $n$")
    a2.set_ylabel(r"Price Change $\Delta p/\Gamma_0$")
    a2.legend(loc="upper left", fontsize=7)
    panel_label(a2, "(a)")
    save(fig, "g05_price_impact", OUT)
    th, Pm = d["g5_theory"], P.mean(axis=0)
    se_end = P[:, T0_MO + N_MO].std(ddof=1) / np.sqrt(len(P))
    return {"GalFiveErr": f"{100 * float(d['g5_err']):.2f}", "GalFivePeakMC": fmt_pm(float(Pm[T0_MO + N_MO]), se_end, 1),
            "GalFivePeakTh": fmt(float(th[T0_MO + N_MO]), 1),
            "GalFiveNoise": fmt(float(P[:, T0_MO + N_MO].std(ddof=1)), 1)}


def g06(d):
    fig = plt.figure(figsize=TALL)
    a1 = fig.add_axes([0.075, 0.17, 0.38, 0.72])
    t = d["g6_t"]
    a1.step(t, d["g6_mid"], where="post", color="0.2", lw=1.1, label="Mid")
    a1.step(t, d["g6_w"], where="post", color=ORANGE, lw=0.6, alpha=0.9, label="Weighted Mid")
    a1.step(t, d["g6_micro"], where="post", color=RED, lw=0.8, label="Micro-Price")
    lo, hi = np.nanmin(d["g6_mid"]), np.nanmax(d["g6_mid"])
    a1.set_ylim(lo - 1.0, hi + 0.5 * (hi - lo) + 1.5)
    a1.set_xlim(0, W_MICRO)
    a1.set_xlabel(r"Time $t$ (s)")
    a1.set_ylabel(r"Price (ticks)")
    a1.legend(loc="upper left", ncol=3, fontsize=7)
    panel_label(a1, "(a)")

    a2 = fig.add_axes([0.47, 0.0, 0.53, 1.0], projection="3d")
    grid, pmf = d["g6_grid"], d["g6_pmf"]
    edges = np.append(grid - 0.25, grid[-1] + 0.25)
    centres = np.arange(0.1, 1.0, 0.2)
    for j in range(5):
        neg, zer = grid < 0, np.isclose(grid, 0)
        iz = int(np.flatnonzero(zer)[0])
        polys = [_bars(edges[:iz + 1], pmf[j, :iz]), _bars(edges[iz:iz + 2], pmf[j, iz:iz + 1]),
                 _bars(edges[iz + 1:], pmf[j, iz + 1:])]
        pc = PolyCollection(polys, facecolors=[BLUE, "0.7", RED], edgecolors="0.15", linewidths=0.3, alpha=0.88)
        a2.add_collection3d(pc, zs=centres[j], zdir="x")
    style_3d(a2, elev=24, azim=-58)
    a2.set_xlim(0, 1)
    a2.set_ylim(grid[0] - 0.25, grid[-1] + 0.25)
    a2.set_yticks([-2, -1, 0, 1, 2])
    a2.set_zlim(0, 0.12 * np.ceil(pmf.max() / 0.12))
    a2.set_box_aspect(None, zoom=0.82)
    a2.set_xlabel(r"Imbalance $I$", labelpad=0)
    a2.set_ylabel(r"Mid Change in 2 s (ticks)", labelpad=0)
    a2.set_zlabel("Probability", labelpad=-1)
    _note3d(a2, "Blue: Down   Red: Up")
    _tag3d(a2, "(b)")
    save(fig, "g06_order_flow", OUT)
    pu = d["g6_pmf"][:, d["g6_grid"] > 0].sum(axis=1)
    pd = d["g6_pmf"][:, d["g6_grid"] < 0].sum(axis=1)
    mdm = d["g6_mean"]
    return {"GalSixUpLow": f"{100 * pu[0]:.0f}", "GalSixUpHigh": f"{100 * pu[-1]:.0f}",
            "GalSixDownLow": f"{100 * pd[0]:.0f}", "GalSixDownHigh": f"{100 * pd[-1]:.0f}",
            "GalSixMeanHigh": fmt_pm(float(mdm[-1, 0]), float(mdm[-1, 1]), 2),
            "GalSixMinBin": fmt_int(int(d["g6_nb"].min()))}


def g07(d):
    fig = plt.figure(figsize=TALL)
    a1 = fig.add_axes([0.47, 0.0, 0.53, 1.0], projection="3d")
    qq = np.arange(-Q_MM, Q_MM)
    Tm, Qm = np.meshgrid(d["g7_t"], qq, indexing="ij")
    Z = d["g7_db"]
    _surface(a1, Tm, Qm, Z, cmap="viridis")
    style_3d(a1, elev=24, azim=-135)
    a1.set_box_aspect(None, zoom=0.82)
    a1.set_xlabel(r"Time $t$ (s)", labelpad=0)
    a1.set_ylabel(r"Inventory $q$ (shares)", labelpad=0)
    a1.set_zlabel(r"Bid Distance $\delta^b$ (ticks)", labelpad=-1)
    _tag3d(a1, "(b)")

    a2 = fig.add_axes([0.075, 0.17, 0.30, 0.72])
    qp, prof = d["g7_q"].astype(float), d["g7_profit"]
    tq = np.linspace(0, P_MM.T, qp.shape[0])
    lo, hi = np.percentile(qp, [5, 95], axis=1)
    a2.fill_between(tq, lo, hi, color="0.85", lw=0, label="5% to 95% of 300 Paths")
    norm = colors.Normalize(*np.percentile(prof, [5, 95]))
    cm = colormaps["plasma"]
    for j in np.argsort(prof)[::7]:
        a2.plot(tq, qp[:, j], color=cm(norm(prof[j])), lw=0.8, alpha=0.9)
    a2.axhline(Q_MM, color="0.3", lw=0.8, ls="--")
    a2.axhline(-Q_MM, color="0.3", lw=0.8, ls="--", label="Inventory Limit")
    a2.legend(loc="upper left", fontsize=7)
    a2.set_xlim(0, P_MM.T)
    a2.set_xlabel(r"Time $t$ (s)")
    a2.set_ylabel(r"Inventory $q_t$ (shares)")
    a2.set_ylim(-Q_MM - 3, Q_MM + 16)
    panel_label(a2, "(a)")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    _cbar(fig, sm, [0.385, 0.17, 0.012, 0.72], "Final Profit (ticks)")
    save(fig, "g07_market_making", OUT)
    q_end = qp[-1]
    return {"GalSevenProfit": fmt_pm(float(prof.mean()), float(prof.std(ddof=1) / np.sqrt(prof.size)), 1),
            "GalSevenProfitSD": fmt(float(prof.std(ddof=1)), 1),
            "GalSevenAbsQ": fmt(float(np.abs(q_end).mean()), 1), "GalSevenPaths": str(prof.size),
            "GalSevenBidZero": fmt(float(Z[0, Q_MM]), 2)}


def g08(d):
    fig = plt.figure(figsize=TALL)
    a1 = fig.add_axes([0.075, 0.17, 0.38, 0.72])
    b, V = d["g8_belief"], d["g8_V"]
    n = np.arange(1, b.shape[1] + 1)
    for j in range(b.shape[0]):
        a1.plot(n, b[j], color=RED if V[j] > 0.5 else BLUE, lw=0.6, alpha=0.15)
    a1.plot(n, d["g8_spread"], color="k", lw=1.6, label="Mean Spread")
    a1.plot([], [], color=RED, label=r"$E[V]$, True $V_H$")
    a1.plot([], [], color=BLUE, label=r"$E[V]$, True $V_L$")
    a1.set_xlim(1, n[-1])
    a1.set_ylim(-0.02, 1.35)
    a1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a1.set_xlabel(r"Trade $n$")
    a1.set_ylabel(r"Price $/(V_H - V_L)$")
    a1.legend(loc="upper center", ncol=3, fontsize=7)
    panel_label(a1, "(a)")

    a2 = fig.add_axes([0.47, 0.0, 0.53, 1.0], projection="3d")
    Pm, Am = np.meshgrid(d["g8_pis"], d["g8_als"], indexing="ij")
    Z = d["g8_surf"]
    a2.computed_zorder = False
    s = _surface(a2, Pm, Am, Z, cmap="magma", norm=colors.Normalize(0, 1))
    s.set_zorder(1)
    j = int(np.argmin(np.abs(d["g8_als"] - ALPHA_GM)))
    ln, = a2.plot(d["g8_pis"], np.full_like(d["g8_pis"], d["g8_als"][j]), Z[:, j], color=GREEN, lw=1.8)
    ln.set_zorder(5)
    style_3d(a2, elev=26, azim=-50)
    a2.set_box_aspect(None, zoom=0.76)
    a2.set_zlim(0, 1)
    a2.set_xlabel(r"Belief $\pi$", labelpad=0)
    a2.set_ylabel(r"Informed Share $\alpha$", labelpad=0)
    a2.set_zlabel(r"Spread $/(V_H - V_L)$", labelpad=-1)
    _note3d(a2, r"Green: $\alpha$ = 0.3, as in (a)")
    _tag3d(a2, "(b)")
    save(fig, "g08_adverse_selection", OUT)
    sp = d["g8_spread"]
    return {"GalEightStart": fmt(float(sp[0]), 2), "GalEightEnd": fmt(float(sp[-1]), 3),
            "GalEightWrong": f"{100 * float(d['g8_wrong']):.1f}"}


def g09(d):
    fig = plt.figure(figsize=TALL)
    a1 = fig.add_axes([0.075, 0.17, 0.38, 0.72])
    t, L = d["g9_t"], d["g9_lvr"].astype(float)
    for j in range(150):
        a1.plot(t, L[:, j], color=BLUE, lw=0.4, alpha=0.2)
    a1.plot(t, L.mean(axis=1), color=RED, lw=1.6, label=f"Mean of {L.shape[1]} Paths")
    a1.plot(t, 100 * (1 - np.exp(-SIGMA_AMM ** 2 * t / 8)), color="k", lw=1.0, ls="--",
            label=r"Exact $1 - e^{-\sigma^2 t/8}$")
    a1.set_xlim(0, T_AMM)
    a1.set_ylim(0, 1.45 * np.percentile(L[-1], 99))
    a1.set_xlabel(r"Time $t$ (days)")
    a1.set_ylabel(r"Cumulative LVR (% of $V_0$)")
    a1.legend(loc="upper left", fontsize=7)
    panel_label(a1, "(a)")

    a2 = fig.add_axes([0.47, 0.0, 0.53, 1.0], projection="3d")
    sig = np.linspace(0.01, 0.10, 37)
    lh = np.linspace(0.0, 3.0, 49)                    # log10 of the half-width in percent
    w = 1 + 10 ** lh / 100
    S, LH = np.meshgrid(sig, lh, indexing="ij")
    Z = np.log10(100 * S ** 2 / (8 * (1 - (1 + 10 ** LH / 100) ** -0.5)))
    a2.computed_zorder = False
    s = _surface(a2, 100 * S, LH, Z, cmap="viridis")
    s.set_zorder(1)
    z0 = float(np.log10(100 * SIGMA_AMM ** 2 / (8 * (1 - 1.1 ** -0.5))))
    a2.scatter([100 * SIGMA_AMM], [1.0], [z0], color=RED, s=30, zorder=6, depthshade=False)
    style_3d(a2, elev=26, azim=-130)
    a2.set_box_aspect(None, zoom=0.82)
    a2.set_yticks([0, 1, 2, 3])
    a2.set_yticklabels(["1", "10", "100", "1000"])
    a2.set_xlabel(r"Volatility $\sigma$ (%/$\sqrt{\mathrm{day}}$)", labelpad=0)
    a2.set_ylabel("Range Half-Width (%)", labelpad=0)
    a2.set_zlabel(r"Log$_{10}$ LVR/V (%/day)", labelpad=-1)
    _note3d(a2, r"Red: $\sigma$ = 4%, Range $\pm$10%")
    _tag3d(a2, "(b)")
    save(fig, "g09_amm", OUT)
    end = L[-1]
    return {"GalNineMean": fmt_pm(float(end.mean()), float(end.std(ddof=1) / np.sqrt(end.size)), 3),
            "GalNineExact": fmt(float(100 * (1 - np.exp(-SIGMA_AMM ** 2 * T_AMM / 8))), 3),
            "GalNineSD": fmt(float(end.std(ddof=1)), 3), "GalNineRange": fmt(10 ** z0, 2)}


FIGS = (g01, g02, g03, g04, g05, g06, g07, g08, g09)


def plot():
    use_style()
    d = load_raw(NAME)
    numbers = {}
    for f in FIGS:
        numbers.update(f(d))
        plt.close("all")
    write_numbers(NAME, numbers)


def main():
    compute()
    plot()


if __name__ == "__main__":
    run(compute, plot)
