"""Markouts of passive fills: how much of the spread a liquidity provider keeps.

A market buy at time t trades against the resting sell order at the front of
the best ask queue. For that passive seller, the markout at horizon h (s) is

    M(h) = p_A(t-) - S(t + h)          (ticks),

the sale price minus the mid price h seconds later. For a passive buyer hit by
a market sell, M(h) = S(t + h) - p_B(t-). At h = 0+ the markout is the realized
half-spread; its decay with h is the adverse selection. The quantities used by
practitioners are

    realized spread(h) = 2 M(h),   price impact(h) = half-spread(t-) - M(h),

both per unit traded. All functions work on a LOBRecord from markov_lob.
"""
from __future__ import annotations

import numpy as np

from .markov_lob import LOBRecord


def passive_fills(rec: LOBRecord):
    """Indices, side and pre-trade state of every market order.

    Returns dict with k (event index), side (+1 when the passive order was a
    sell, -1 for a passive buy), price (passive fill price, ticks), q_before
    (size of the hit queue before the trade), imb_before (q_bid / (q_bid + q_ask)
    before the trade, in [0, 1]), and spread_before (ticks).
    """
    k = np.flatnonzero((rec.event == 2) | (rec.event == 3))
    k = k[k > 0]
    prev = k - 1
    side = np.where(rec.event[k] == 2, 1, -1)
    ok = (rec.q_bid[prev] > 0) & (rec.q_ask[prev] > 0)
    k, prev, side = k[ok], prev[ok], side[ok]
    price = np.where(side > 0, rec.best_ask[prev], rec.best_bid[prev]).astype(float)
    q_before = np.where(side > 0, rec.q_ask[prev], rec.q_bid[prev])
    qb, qa = rec.q_bid[prev].astype(float), rec.q_ask[prev].astype(float)
    return {"k": k, "side": side, "price": price, "q_before": q_before,
            "imb_before": qb / (qb + qa),
            "spread_before": (rec.best_ask[prev] - rec.best_bid[prev]).astype(float)}


def mid_at(rec: LOBRecord, times):
    """Mid price (ticks) in force at each of the given times."""
    mid = 0.5 * (rec.best_bid + rec.best_ask)
    idx = np.searchsorted(rec.t, times, side="right") - 1
    return mid[np.clip(idx, 0, len(mid) - 1)]


def markout_matrix(rec: LOBRecord, horizons):
    """Markout M(h) (ticks) for every passive fill and horizon, shape (n_fills, n_h).

    h = 0 means just after the trade. Fills whose horizon runs past the end of
    the record are set to NaN.
    """
    f = passive_fills(rec)
    t0 = rec.t[f["k"]]
    out = np.empty((len(t0), len(horizons)))
    for j, h in enumerate(horizons):
        S = mid_at(rec, t0 + h)
        m = f["side"] * (f["price"] - S)
        m[t0 + h > rec.t[-1]] = np.nan
        out[:, j] = m
    return f, out


def mean_with_se(x, axis=0):
    """Column means and standard errors, ignoring NaN. Treats fills as independent.

    Overlapping horizons make nearby fills correlated, so these standard errors
    are optimistic for long horizons; use blocks of the record for a check.
    """
    n = np.sum(np.isfinite(x), axis=axis)
    return np.nanmean(x, axis=axis), np.nanstd(x, axis=axis, ddof=1) / np.sqrt(n), n
