"""Adverse selection in quote setting: the Glosten-Milgrom (1985) model.

The asset is worth V_H or V_L (price units), with prior P(V = V_H) = pi.
Each period one trader arrives. With probability alpha (dimensionless) the
trader is informed and buys if V = V_H, sells if V = V_L. Otherwise the
trader is uninformed and buys or sells with probability 1/2 each. A
competitive, risk-neutral market maker earns zero expected profit, so it
quotes

    ask = E[V | buy],   bid = E[V | sell].

With P(buy | H) = alpha + (1 - alpha)/2 and P(buy | L) = (1 - alpha)/2, the
spread at pi = 1/2 is alpha (V_H - V_L). After each trade the market maker
updates pi by Bayes' rule, so the quotes converge to the true value and the
spread closes.

For the LP this is the markout identity in its cleanest form: the expected
markout against the true value is zero; the spread earned from uninformed
traders exactly pays for the losses to informed ones.
"""
from __future__ import annotations

import numpy as np


def quotes(pi, alpha, v_low=0.0, v_high=1.0):
    """Zero-profit bid and ask (price) at belief pi = P(V = V_H)."""
    pi = np.asarray(pi, float)
    pb_h = alpha + 0.5 * (1 - alpha)       # P(buy | H)
    pb_l = 0.5 * (1 - alpha)               # P(buy | L)
    post_buy = pi * pb_h / (pi * pb_h + (1 - pi) * pb_l)
    post_sell = pi * pb_l / (pi * pb_l + (1 - pi) * pb_h)
    ask = v_low + (v_high - v_low) * post_buy
    bid = v_low + (v_high - v_low) * post_sell
    return bid, ask, post_buy, post_sell


def simulate(alpha, n_trades, n_paths, v_low=0.0, v_high=1.0, pi0=0.5, rng=None):
    """Simulate the sequential trading game.

    Returns dict of (n_paths, n_trades) arrays: 'spread' (price), 'belief'
    (pi before each trade), 'eps' (+1 when the market maker sells, that is
    the trader buys), 'price' (trade price), 'informed' (bool), and the true
    value per path 'V'.
    """
    rng = np.random.default_rng(rng)
    V_high = rng.random(n_paths) < pi0
    V = np.where(V_high, v_high, v_low)
    pi = np.full(n_paths, pi0)
    out = {k: np.empty((n_paths, n_trades)) for k in ("spread", "belief", "eps", "price")}
    out["informed"] = np.empty((n_paths, n_trades), dtype=bool)
    for n in range(n_trades):
        bid, ask, pb, ps = quotes(pi, alpha, v_low, v_high)
        informed = rng.random(n_paths) < alpha
        buy = np.where(informed, V_high, rng.random(n_paths) < 0.5)
        out["spread"][:, n] = ask - bid
        out["belief"][:, n] = pi
        out["eps"][:, n] = np.where(buy, 1.0, -1.0)
        out["price"][:, n] = np.where(buy, ask, bid)
        out["informed"][:, n] = informed
        pi = np.where(buy, pb, ps)
    out["V"] = V
    return out


def markouts_vs_value(sim):
    """Market maker's markout per trade against the true value, eps (p - V) (price).

    Returns (all, uninformed, informed) means over all trades in the sample.
    """
    m = sim["eps"] * (sim["price"] - sim["V"][:, None])
    inf = sim["informed"]
    return float(m.mean()), float(m[~inf].mean()), float(m[inf].mean())
