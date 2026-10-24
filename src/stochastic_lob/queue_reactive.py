"""Queue-reactive model (Huang, Lehalle and Rosenbaum 2015), Model I.

Each price level holds a queue of q >= 0 orders (in units of an average event
size). In Model I the queues are independent, and each is a birth-death
process whose rates depend on its own size q:

    q -> q + 1   rate lam_L(q)                  limit order insertion (1/s)
    q -> q - 1   rate lam_C(q) + lam_M(q)       cancellation + market order (1/s)

with lam_C(0) = lam_M(0) = 0. If sum_q pi(q) is finite, the stationary law is

    pi(q) = pi(0) prod_{k=1}^{q} lam_L(k-1) / (lam_C(k) + lam_M(k)),

the product formula for birth-death chains in detailed balance.

HLR also define Models II and III. In Model III the reference price p_ref
moves with probability theta when a best queue empties, and with probability
theta_reinit the whole book is then redrawn from its stationary law (news).
simulate_model3 implements a reduced version with two levels per side; the
reductions are listed in its docstring.

Estimation (HLR Section 2.3.2). For a queue observed at size n for a total
time tau(n) (s), during which N_L(n), N_C(n), N_M(n) events of each type
occur, the maximum-likelihood intensities are N_x(n) / tau(n), with Poisson
standard errors sqrt(N_x(n)) / tau(n). This is the HLR estimator
Lambda(n) * (share of type x), written in exposure form.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

RateFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class QueueRates:
    """Intensity functions of one queue, each mapping queue size q to a rate (1/s)."""

    limit: RateFn
    cancel: RateFn
    market: RateFn

    def up(self, q):
        return self.limit(np.asarray(q))

    def down(self, q):
        q = np.asarray(q)
        return np.where(q > 0, self.cancel(q) + self.market(q), 0.0)


def illustrative_rates(lam0=1.2, q_half=4.0, lam_inf=0.3, c=0.15, m=0.4) -> QueueRates:
    """A queue-reactive rate set, for illustration only (not fitted to data).

    Limit orders arrive faster at short queues and slow down as the queue
    grows; each order cancels at rate c; market orders arrive at rate m while
    the queue is non-empty.

        lam_L(q) = lam_inf + (lam0 - lam_inf) exp(-q / q_half)
        lam_C(q) = c q
        lam_M(q) = m 1{q > 0}
    """
    return QueueRates(
        limit=lambda q: lam_inf + (lam0 - lam_inf) * np.exp(-np.asarray(q, float) / q_half),
        cancel=lambda q: c * np.asarray(q, float),
        market=lambda q: m * (np.asarray(q) > 0),
    )


def stationary_distribution(rates: QueueRates, q_max: int) -> np.ndarray:
    """Stationary law pi(q), q = 0..q_max, from the birth-death product formula.

    The tail beyond q_max is dropped and the result renormalized. Check that
    pi(q_max) is negligible before trusting the truncation.
    """
    q = np.arange(1, q_max + 1)
    log_ratio = np.log(rates.up(q - 1)) - np.log(rates.down(q))
    log_pi = np.concatenate([[0.0], np.cumsum(log_ratio)])
    pi = np.exp(log_pi - log_pi.max())
    return pi / pi.sum()


def simulate_queue(rates: QueueRates, t_end: float, q0: int = 0, rng=None, q_max: int = 10_000):
    """Exact simulation of one queue up to time t_end (s).

    Returns (occupation, n_events) where occupation[q] is the fraction of time
    spent at size q.
    """
    rng = np.random.default_rng(rng)
    occ = np.zeros(q_max + 1)
    q = q0
    t = 0.0
    n = 0
    while t < t_end:
        r_up = float(rates.up(q))
        r_dn = float(rates.down(q))
        R = r_up + r_dn
        dt = min(rng.exponential(1.0 / R), t_end - t)
        occ[q] += dt
        t += dt
        if t >= t_end:
            break
        q = q + 1 if rng.random() * R < r_up else q - 1
        n += 1
        if q > q_max:
            raise RuntimeError("queue exceeded q_max; the chain may be transient")
    return occ / t_end, n


def mean_queue_size(pi: np.ndarray) -> float:
    return float(np.arange(len(pi)) @ pi)


# ---------------------------------------------------------------------- estimation
def simulate_queue_events(rates: QueueRates, t_end: float, q0: int = 0, rng=None):
    """Simulate one queue and record every event.

    Returns (wait, q_before, kind, censored): the time (s) since the previous
    event, the queue size before the event, kind 0 = limit, 1 = cancel,
    2 = market, and the censored last spell as (q_last, time_left).
    """
    rng = np.random.default_rng(rng)
    q, t = q0, 0.0
    waits, qs, kinds = [], [], []
    while True:
        lam_l = float(rates.limit(np.asarray(q)))
        lam_c = float(rates.cancel(np.asarray(q))) if q > 0 else 0.0
        lam_m = float(rates.market(np.asarray(q))) if q > 0 else 0.0
        R = lam_l + lam_c + lam_m
        dt = rng.exponential(1.0 / R)
        if t + dt > t_end:
            return (np.asarray(waits), np.asarray(qs, dtype=np.int64),
                    np.asarray(kinds, dtype=np.int64), (q, t_end - t))
        t += dt
        u = rng.random() * R
        kind = 0 if u < lam_l else (1 if u < lam_l + lam_c else 2)
        waits.append(dt)
        qs.append(q)
        kinds.append(kind)
        q = q + 1 if kind == 0 else q - 1


def estimate_intensities(wait, q_before, kind, n_max, censored=None):
    """MLE of the limit, cancel and market intensities at queue sizes n = 0..n_max.

    Returns dict with arrays 'limit', 'cancel', 'market' (1/s), their standard
    errors 'se_limit' etc., the exposure 'tau' (s) and the event counts 'count'.
    Sizes with no exposure get NaN. Events at sizes above n_max are ignored.
    """
    keep = q_before <= n_max
    wait, q_before, kind = wait[keep], q_before[keep], kind[keep]
    tau = np.bincount(q_before, weights=wait, minlength=n_max + 1).astype(float)
    if censored is not None and censored[0] <= n_max:
        tau[censored[0]] += censored[1]
    out = {"tau": tau, "count": np.bincount(q_before, minlength=n_max + 1)}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k, name in enumerate(("limit", "cancel", "market")):
            N = np.bincount(q_before[kind == k], minlength=n_max + 1).astype(float)
            out[name] = np.where(tau > 0, N / tau, np.nan)
            out["se_" + name] = np.where(tau > 0, np.sqrt(N) / tau, np.nan)
    return out


# ---------------------------------------------------------------------- Model III (reduced)
@dataclass
class Model3Params:
    """Reduced queue-reactive Model III with two levels per side.

    first, second: intensity functions at the best level and at the second level.
    theta: probability that p_ref moves when a best queue empties.
    theta_reinit: probability that, after a move, the book is redrawn from the
    stationary laws of Model I (exogenous information).
    """

    first: QueueRates
    second: QueueRates
    theta: float = 0.7
    theta_reinit: float = 0.85
    q_max: int = 80


def hlr_like_rates():
    """Illustrative intensities with the shapes HLR report for a large-tick stock.

    Best level: insertion flat in q but lower at q = 0, cancellation concave
    in q, market orders decaying exponentially in q. Second level: insertion
    decreasing in q and lower cancellation. Market orders reach the second
    level only when the best level is empty (handled in simulate_model3).
    Not fitted to data.
    """
    first = QueueRates(
        limit=lambda q: np.where(np.asarray(q) == 0, 0.6, 1.0) + 0.0 * np.asarray(q, float),
        cancel=lambda q: 1.2 * (1 - np.exp(-np.asarray(q, float) / 6.0)),
        market=lambda q: (np.asarray(q) > 0) * (0.2 + 0.6 * np.exp(-np.asarray(q, float) / 3.0)),
    )
    second = QueueRates(
        limit=lambda q: 0.3 + 0.7 * np.exp(-np.asarray(q, float) / 5.0),
        cancel=lambda q: 0.6 * (1 - np.exp(-np.asarray(q, float) / 4.0)),
        market=lambda q: (np.asarray(q) > 0) * (0.2 + 0.6 * np.exp(-np.asarray(q, float) / 3.0)),
    )
    return first, second


def simulate_model3(p: Model3Params, t_end: float, rng=None, q0=None):
    """Simulate the reduced Model III. Returns (move_times, move_signs, n_events).

    State: queues (b2, b1, a1, a2) at p_ref - 1.5, - 0.5, + 0.5, + 1.5 ticks.
    Each queue follows its Model I intensities, except that market orders
    reach a second-level queue only while the best queue on its side is empty.
    When a best queue goes from 1 to 0, p_ref moves one tick away from it with
    probability theta. The queues then shift by one level: the emptied level
    becomes the new best queue on the other side (size 0), and the new second
    level is drawn from its stationary law. With probability theta_reinit
    (given a move) all four queues are redrawn from the stationary laws.

    Reductions relative to HLR: two levels instead of three, Model I dynamics
    between price moves, unit event sizes, and moves triggered only by
    depletions (not by insertions inside a wide spread).
    """
    rng = np.random.default_rng(rng)
    pi1 = stationary_distribution(p.first, p.q_max)
    pi2 = stationary_distribution(p.second, p.q_max)

    def draw(pi):
        return int(rng.choice(len(pi), p=pi))

    if q0 is None:
        b2, b1, a1, a2 = draw(pi2), draw(pi1), draw(pi1), draw(pi2)
    else:
        b2, b1, a1, a2 = q0
    # Rate tables indexed by queue size; queues never get near the table end
    # for intensities whose stationary law is negligible beyond q_max.
    qg = np.arange(10 * p.q_max + 1)
    Lf, Cf, Mf = p.first.limit(qg), p.first.cancel(qg), p.first.market(qg)
    L2, C2, M2 = p.second.limit(qg), p.second.cancel(qg), p.second.market(qg)
    t = 0.0
    times, signs = [], []
    n_ev = 0
    while True:
        lim = [L2[b2], Lf[b1], Lf[a1], L2[a2]]
        dn = [
            (C2[b2] + (M2[b2] if b1 == 0 else 0.0)) if b2 > 0 else 0.0,
            Cf[b1] + Mf[b1] if b1 > 0 else 0.0,
            Cf[a1] + Mf[a1] if a1 > 0 else 0.0,
            (C2[a2] + (M2[a2] if a1 == 0 else 0.0)) if a2 > 0 else 0.0,
        ]
        rates = np.array(lim + dn)
        R = rates.sum()
        t += rng.exponential(1.0 / R)
        if t > t_end:
            return np.asarray(times), np.asarray(signs, dtype=np.int64), n_ev
        n_ev += 1
        e = min(int(np.searchsorted(np.cumsum(rates), rng.random() * R, side="right")), 7)
        qq = [b2, b1, a1, a2]
        if e < 4:
            qq[e] += 1
        else:
            qq[e - 4] -= 1
        b2, b1, a1, a2 = qq
        depleted = 0
        if e == 5 and b1 == 0:
            depleted = -1             # best bid emptied: price may move down
        elif e == 6 and a1 == 0:
            depleted = +1             # best ask emptied: price may move up
        if depleted and rng.random() < p.theta:
            times.append(t)
            signs.append(depleted)
            if rng.random() < p.theta_reinit:
                b2, b1, a1, a2 = draw(pi2), draw(pi1), draw(pi1), draw(pi2)
            elif depleted > 0:
                b2, b1, a1, a2 = b1, 0, a2, draw(pi2)
            else:
                b2, b1, a1, a2 = draw(pi2), b2, 0, a1


def mean_reversion_ratio(signs) -> float:
    """eta = N_c / (2 N_a) (Robert and Rosenbaum): continuations over twice the alternations.

    eta = 1/2 for independent symmetric moves; eta < 1/2 means mean reversion.
    """
    s = np.asarray(signs)
    same = np.sum(s[1:] == s[:-1])
    alt = np.sum(s[1:] != s[:-1])
    return float(same / (2.0 * alt)) if alt else np.inf


def realized_vol(times, signs, t_end, horizon):
    """Standard deviation (ticks) of p_ref changes over consecutive windows of length horizon (s)."""
    edges = np.arange(0.0, t_end + 1e-9, horizon)
    path = np.concatenate([[0], np.cumsum(signs)])
    idx = np.searchsorted(np.asarray(times), edges, side="right")
    return float(np.std(np.diff(path[idx]), ddof=1))
