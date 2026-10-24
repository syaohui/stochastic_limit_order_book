"""The micro-price of Stoikov (2018), estimated from a simulated book.

The state is x = (I, s): the queue imbalance I = q_b / (q_b + q_a), cut into
n_imb buckets, and the spread s (ticks). The micro-price is the limit of the
expected mid price after the next k mid-price changes,

    P_micro = M + g_1(x) + g_2(x) + ...,   g_{i+1} = B g_i,

where M is the mid (ticks). From consecutive observations, with dM the mid
change between them:

    Q(x, y) = P(next state y, dM = 0 | x)     (moves that keep the mid)
    T(x, y) = P(next state y, dM != 0 | x)    (moves that change the mid)
    R(x, k) = P(dM = k | x) for k != 0,

    g_1 = (I - Q)^{-1} R K,   B = (I - Q)^{-1} T,

with K the vector of possible mid changes. The data are symmetrized (each
observation also enters mirrored, I -> 1 - I and dM -> -dM) so that the
adjustment is odd in the imbalance and the series converges.

The book here is the Markovian model of markov_lob, observed after every
event. Stoikov estimates the same objects from real quotes.
"""
from __future__ import annotations

import numpy as np


def states(q_bid, q_ask, spread, n_imb=10, max_spread=2):
    """State index (imbalance bucket, spread) or -1 when outside the grid."""
    q_bid = np.asarray(q_bid, float)
    q_ask = np.asarray(q_ask, float)
    tot = q_bid + q_ask
    with np.errstate(invalid="ignore", divide="ignore"):
        imb = np.where(tot > 0, q_bid / tot, np.nan)
    ib = np.clip(np.floor(imb * n_imb), 0, n_imb - 1)
    ok = np.isfinite(imb) & (spread >= 1) & (spread <= max_spread) & (q_bid > 0) & (q_ask > 0)
    idx = np.where(ok, (np.asarray(spread) - 1) * n_imb + ib, -1)
    return idx.astype(int), imb


def mirror(idx, n_imb):
    """Index of the mirrored state (I -> 1 - I, same spread)."""
    s, i = np.divmod(idx, n_imb)
    return s * n_imb + (n_imb - 1 - i)


def fit(idx, mid, n_imb=10, max_spread=2, n_terms=20):
    """Estimate g_1, B and the micro-price adjustment G (ticks) per state.

    idx: state index per observation (-1 to skip); mid: mid price (ticks).
    Transitions are taken between consecutive valid observations.
    Returns dict with 'G' (total adjustment), 'g' (list of g_i), 'B', 'n'
    (visits per state) and 'K' (mid-change values).
    """
    idx = np.asarray(idx)
    mid = np.asarray(mid, float)
    ok = idx >= 0
    x0, x1 = idx[:-1][ok[:-1] & ok[1:]], idx[1:][ok[:-1] & ok[1:]]
    dm = np.round((mid[1:] - mid[:-1])[ok[:-1] & ok[1:]] * 2) / 2
    # symmetrize
    x0 = np.concatenate([x0, mirror(x0, n_imb)])
    x1 = np.concatenate([x1, mirror(x1, n_imb)])
    dm = np.concatenate([dm, -dm])
    S = n_imb * max_spread
    K = np.unique(dm[dm != 0])
    n = np.bincount(x0, minlength=S).astype(float)
    Q = np.zeros((S, S))
    T = np.zeros((S, S))
    R = np.zeros((S, len(K)))
    same = dm == 0
    np.add.at(Q, (x0[same], x1[same]), 1.0)
    np.add.at(T, (x0[~same], x1[~same]), 1.0)
    np.add.at(R, (x0[~same], np.searchsorted(K, dm[~same])), 1.0)
    seen = n > 0
    Q[seen] /= n[seen, None]
    T[seen] /= n[seen, None]
    R[seen] /= n[seen, None]
    # States never visited are removed from the linear algebra.
    keep = np.flatnonzero(seen)
    Qk, Tk, Rk = Q[np.ix_(keep, keep)], T[np.ix_(keep, keep)], R[keep]
    inv = np.linalg.inv(np.eye(len(keep)) - Qk)
    g1 = inv @ Rk @ K
    B = inv @ Tk
    gs = [g1]
    for _ in range(n_terms - 1):
        gs.append(B @ gs[-1])
    G = np.full(S, np.nan)
    G[keep] = np.sum(gs, axis=0)
    g_full = []
    for g in gs:
        v = np.full(S, np.nan)
        v[keep] = g
        g_full.append(v)
    return {"G": G, "g": g_full, "B": B, "n": n, "K": K}


def predictors(idx, imb, mid, spread, G):
    """Mid, weighted mid and micro-price (ticks) at each observation.

    Weighted mid = I p_A + (1 - I) p_B = mid + (I - 1/2) s.
    The micro-price is NaN where the state is invalid or unseen in training.
    """
    w = mid + (imb - 0.5) * spread
    micro = np.where(idx >= 0, mid + G[np.clip(idx, 0, None)], np.nan)
    return mid, w, micro
