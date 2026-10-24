"""Liquidity provision in an automated market maker: loss-versus-rebalancing.

A constant-product market maker (CPMM) holds x units of a risky asset and y
units of the numeraire with x y = L^2, where L is the liquidity (units of
sqrt(asset * numeraire)). Its marginal price is P = y / x, so

    x(P) = L / sqrt(P),   y(P) = L sqrt(P),   V(P) = x P + y = 2 L sqrt(P).

The reference price on a liquid exchange follows dP = sigma P dW (geometric
Brownian motion, sigma in 1/sqrt(time)). Arbitrageurs keep the pool price
equal to P. Milionis, Moallemi, Roughgarden and Zhang (2022) compare the LP
with a rebalancing strategy that holds the same x(P_t) but trades at the
exchange price. The gap, loss-versus-rebalancing (LVR), accrues at rate

    dLVR/dt = (sigma^2 P^2 / 2) |x'(P)| = sigma^2 V(P) / 8      (CPMM).

Because P is a martingale and E[sqrt(P_T)] = sqrt(P_0) exp(-sigma^2 T / 8),
the expected LVR over [0, T] is exactly V_0 (1 - exp(-sigma^2 T / 8)).

Concentrated liquidity on [P_a, P_b] (Uniswap v3) uses
x(P) = L (1/sqrt(P) - 1/sqrt(P_b)), y(P) = L (sqrt(P) - sqrt(P_a)) inside the
range. The LVR rate there is sigma^2 L sqrt(P) / 4, as for the CPMM with the
same L, but the capital is smaller. For a range P_a = P/w, P_b = P w with
w > 1 centred (in log price) on the current price,

    LVR / V = sigma^2 / (8 (1 - w^(-1/2)))   per unit time,

so concentration multiplies the LVR per unit of capital by 1/(1 - w^(-1/2)).
This last formula is derived here from the LVR rate above; it holds only while
the price is inside the range.

With a proportional fee f and arbitrage only at discrete block times, the
pool price can deviate from P inside a no-trade band, and arbitrage profit
falls (Milionis, Moallemi and Roughgarden 2023). simulate_cpmm_arbitrage
measures this by simulation.
"""
from __future__ import annotations

import numpy as np


def cpmm_value(P, L=1.0):
    return 2.0 * L * np.sqrt(P)


def lvr_rate_cpmm(P, sigma, L=1.0):
    """Instantaneous LVR rate sigma^2 V / 8 (numeraire per unit time)."""
    return sigma ** 2 * cpmm_value(P, L) / 8.0


def expected_lvr_cpmm(T, sigma, P0=1.0, L=1.0):
    """Exact E[LVR_T] = V_0 (1 - exp(-sigma^2 T / 8)) for driftless GBM."""
    return cpmm_value(P0, L) * (1.0 - np.exp(-sigma ** 2 * T / 8.0))


def lvr_per_capital_concentrated(sigma, w):
    """LVR / V per unit time for a range [P/w, P w] centred on the price."""
    return sigma ** 2 / (8.0 * (1.0 - w ** -0.5))


def simulate_lvr_cpmm(sigma, T, n_steps, n_paths, P0=1.0, L=1.0, rng=None):
    """LVR_T = R_T - V_T on simulated paths, with continuous (every-step) arbitrage.

    R is the rebalancing strategy's value: R_T = V_0 + sum x(P_i) (P_{i+1} - P_i).
    Returns an array of LVR_T, one per path.
    """
    rng = np.random.default_rng(rng)
    dt = T / n_steps
    logP = np.full(n_paths, np.log(P0))
    R = np.full(n_paths, cpmm_value(P0, L))
    for _ in range(n_steps):
        P_old = np.exp(logP)
        logP += -0.5 * sigma ** 2 * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_paths)
        R += L / np.sqrt(P_old) * (np.exp(logP) - P_old)
    return R - cpmm_value(np.exp(logP), L)


def simulate_cpmm_arbitrage(sigma, T, block_dt, fee, n_paths, P0=1.0, L=1.0, rng=None):
    """Arbitrage profit and LP fee income for a CPMM with fee f and discrete blocks.

    At each block the arbitrageur trades only if the exchange price P leaves
    the band [P_pool / (1 + f), P_pool (1 + f)], and moves the pool price to the
    edge of the band. The fee is charged on the input amount and paid to the
    LP outside the pool. Returns dict with total arbitrage profit and total fee
    income per path, both valued at the exchange price at trade time.
    """
    rng = np.random.default_rng(rng)
    n_blocks = int(round(T / block_dt))
    P = np.full(n_paths, P0)
    Pp = np.full(n_paths, P0)
    arb = np.zeros(n_paths)
    fees = np.zeros(n_paths)
    for _ in range(n_blocks):
        P *= np.exp(-0.5 * sigma ** 2 * block_dt + sigma * np.sqrt(block_dt) * rng.standard_normal(n_paths))
        x, y = L / np.sqrt(Pp), L * np.sqrt(Pp)
        up = P > Pp * (1 + fee)
        dn = P < Pp / (1 + fee)
        target = np.where(up, P / (1 + fee), np.where(dn, P * (1 + fee), Pp))
        x2, y2 = L / np.sqrt(target), L * np.sqrt(target)
        # Up: arbitrageur buys x - x2 from the pool, pays (y2 - y)(1 + f).
        prof_up = (x - x2) * P - (1 + fee) * (y2 - y)
        fee_up = fee * (y2 - y)
        # Down: arbitrageur sells x2 - x to the pool, pays (x2 - x)(1 + f), gets y - y2.
        prof_dn = (y - y2) - (1 + fee) * (x2 - x) * P
        fee_dn = fee * (x2 - x) * P
        arb += np.where(up, prof_up, 0.0) + np.where(dn, prof_dn, 0.0)
        fees += np.where(up, fee_up, 0.0) + np.where(dn, fee_dn, 0.0)
        Pp = target
    return {"arb": arb, "fees": fees}


def range_position(P, Pa, Pb, L=1.0):
    """Holdings (x, y) and value of a concentrated position on [Pa, Pb] at price P."""
    sP = np.sqrt(np.clip(P, Pa, Pb))
    x = L * (1.0 / sP - 1.0 / np.sqrt(Pb))
    y = L * (sP - np.sqrt(Pa))
    return x, y, x * P + y


def simulate_lvr_range(sigma, T, n_steps, n_paths, Pa, Pb, P0=1.0, L=1.0, rng=None):
    """LVR_T = R_T - V_T for a concentrated position, as in simulate_lvr_cpmm.

    Also returns the fraction of time the price spent inside [Pa, Pb]; the
    closed-form LVR rate applies only inside the range.
    """
    rng = np.random.default_rng(rng)
    dt = T / n_steps
    logP = np.full(n_paths, np.log(P0))
    R = range_position(P0, Pa, Pb, L)[2] * np.ones(n_paths)
    inside = np.zeros(n_paths)
    for _ in range(n_steps):
        P_old = np.exp(logP)
        x_old = range_position(P_old, Pa, Pb, L)[0]
        inside += (P_old > Pa) & (P_old < Pb)
        logP += -0.5 * sigma ** 2 * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_paths)
        R += x_old * (np.exp(logP) - P_old)
    return R - range_position(np.exp(logP), Pa, Pb, L)[2], inside / n_steps
