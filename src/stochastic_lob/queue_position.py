"""Value of queue position for a resting limit order.

Setting: a large-tick book with a one-tick spread, prices in ticks. We hold one
unit sell order at the best ask. Around it the state is

    a  orders ahead of us in the ask queue (FIFO priority)
    b  orders behind us in the ask queue
    c  orders in the best bid queue

and each queue follows the Markovian rules of Cont, Stoikov and Talreja:

    market buy            rate mu        removes the order at the FRONT of the ask queue
    limit sell joins      rate lam       b -> b + 1 (joins at the back)
    cancel ahead          rate theta*a   a -> a - 1
    cancel behind         rate theta*b   b -> b - 1
    limit buy joins bid   rate lam       c -> c + 1
    market sell or cancel rate mu + theta*c   c -> c - 1
    informed buy sweep    rate nu        takes the whole ask queue (we are filled)
    informed sell sweep   rate nu        takes the whole bid queue (price moves away)

Our order is filled when a = 0 and a market buy arrives. If the bid queue
empties first, the price ticks down and our order is no longer at the best
quote ("price moved away"); we then cancel it at zero value.

Adverse selection. Measure the trade against the mid price just after the NEXT
price change (a one-change markout). We sold at p_A, so with the mid at fill
time m0 = p_A - 1/2:

  * if b = 0 at the fill, our fill empties the queue and the mid jumps to
    p_A + 1/2 at once: markout -1/2 tick;
  * if b > 0, the next change is up with probability P_up(c, b), giving
    markout -1/2, and down otherwise, giving +3/2.

So E[markout | fill] = 1/2 - AS, with adverse selection AS = 1 for b = 0 and
AS = 2 P_up(c, b) - 1 for b > 0. An informed sweep fills every order in the
queue and moves the fair value up by J ticks, so its fill has AS = J.

Orders near the front are mostly filled by small uninformed orders; orders near
the back are filled more often by sweeps or by the trade that empties the
queue. This is the economic reason queue position has value (Moallemi and
Yuan 2017). With nu = 0 the model has no informed flow, and adverse selection
comes from queue mechanics alone.

All quantities are solved exactly on a truncated state space by sparse linear
algebra. The truncation reflects b at b_max and c at c_max; check that results
do not move when these are raised.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


@dataclass
class QueueRates:
    lam: float = 1.0     # limit order arrival rate at each best queue (1/s)
    mu: float = 0.6      # market order rate per side (1/s)
    theta: float = 0.04  # cancellation rate per resting order (1/s)
    nu: float = 0.0      # informed sweep rate per side (1/s)
    jump: float = 2.0    # fair-value move after a sweep, J (ticks)


def p_up_state_dependent(r: QueueRates, b_max: int, c_max: int) -> np.ndarray:
    """P(ask queue empties before bid queue) for CST-type best queues.

    Each queue grows at rate lam and shrinks at rate mu + theta*q. Returns
    h[c, b] for 0 <= c <= c_max, 0 <= b <= b_max, with h[c, 0] = 1 and
    h[0, b] = 0 (h[0, 0] is set to 0.5 and never used).
    """
    nb, nc = b_max, c_max
    idx = lambda c, b: (c - 1) * nb + (b - 1)          # noqa: E731
    rows, cols, vals = [], [], []
    rhs = np.zeros(nb * nc)
    for c in range(1, nc + 1):
        for b in range(1, nb + 1):
            i = idx(c, b)
            moves = [((c + 1, b), r.lam if c < nc else 0.0),
                     ((c - 1, b), r.mu + r.theta * c),
                     ((c, b + 1), r.lam if b < nb else 0.0),
                     ((c, b - 1), r.mu + r.theta * b)]
            tot = sum(w for _, w in moves)
            rows.append(i); cols.append(i); vals.append(tot)
            for (c2, b2), w in moves:
                if w == 0.0:
                    continue
                if b2 == 0:
                    rhs[i] += w                          # ask empties: up move
                elif c2 == 0:
                    pass                                 # bid empties: down move, h = 0
                else:
                    rows.append(i); cols.append(idx(c2, b2)); vals.append(-w)
    A = coo_matrix((vals, (rows, cols)), shape=(nb * nc, nb * nc)).tocsr()
    h = spsolve(A, rhs).reshape(nc, nb)
    out = np.zeros((nc + 1, nb + 1))
    out[1:, 1:] = h
    out[1:, 0] = 1.0
    out[0, 0] = 0.5
    return out


@dataclass
class QueueValue:
    p_fill: np.ndarray       # P(fill) at (a, b, c)
    value: np.ndarray        # E[markout * 1{fill}] at (a, b, c), ticks
    adverse: np.ndarray      # E[AS * 1{fill}] at (a, b, c), ticks
    p_up: np.ndarray         # P_up(c, b) table used for the terminal payoff


def solve_queue_value(r: QueueRates, a_max: int, b_max: int = 60, c_max: int = 60) -> QueueValue:
    """Exact fill probability and expected markout for every (a, b, c).

    Arrays are indexed [a, b, c] with 0 <= a <= a_max, 0 <= b <= b_max,
    0 <= c <= c_max (c = 0 entries are zero: the price has moved away).
    """
    pup = p_up_state_dependent(r, b_max, c_max)

    def as_cost(b, c):
        return 1.0 if b == 0 else 2.0 * pup[c, b] - 1.0

    na, nb, nc = a_max + 1, b_max + 1, c_max
    n = na * nb * nc
    idx = lambda a, b, c: (a * nb + b) * nc + (c - 1)   # noqa: E731
    rows, cols, vals = [], [], []
    r_fill = np.zeros(n)
    r_val = np.zeros(n)
    r_as = np.zeros(n)
    for a in range(na):
        for b in range(nb):
            for c in range(1, nc + 1):
                i = idx(a, b, c)
                moves = [((a, b + 1, c), r.lam if b < b_max else 0.0),
                         ((a, b - 1, c), r.theta * b),
                         ((a - 1, b, c), r.theta * a),
                         ((a, b, c + 1), r.lam if c < c_max else 0.0),
                         ((a, b, c - 1), r.mu + r.theta * c)]
                tot = sum(w for _, w in moves) + r.mu + 2.0 * r.nu
                rows.append(i); cols.append(i); vals.append(tot)
                # Informed buy sweep: filled with AS = J. Sell sweep: price moves away.
                r_fill[i] += r.nu
                r_val[i] += r.nu * (0.5 - r.jump)
                r_as[i] += r.nu * r.jump
                # Market buy: fills us if a = 0, else consumes the front order.
                if a == 0:
                    r_fill[i] += r.mu
                    cost = as_cost(b, c)
                    r_val[i] += r.mu * (0.5 - cost)
                    r_as[i] += r.mu * cost
                else:
                    rows.append(i); cols.append(idx(a - 1, b, c)); vals.append(-r.mu)
                for (a2, b2, c2), w in moves:
                    if w == 0.0 or c2 == 0:
                        continue                         # c2 = 0: price moved away, value 0
                    rows.append(i); cols.append(idx(a2, b2, c2)); vals.append(-w)
    A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    sol = spsolve(A, np.column_stack([r_fill, r_val, r_as]))

    def shape(v):
        out = np.zeros((na, nb, nc + 1))
        out[:, :, 1:] = v.reshape(na, nb, nc)
        return out

    return QueueValue(p_fill=shape(sol[:, 0]), value=shape(sol[:, 1]),
                      adverse=shape(sol[:, 2]), p_up=pup)


def simulate_order(r: QueueRates, a: int, b: int, c: int, n_paths: int, rng=None, pup=None):
    """Monte Carlo of one resting order. Returns (filled, b_at_fill, c_at_fill) arrays.

    Used to test solve_queue_value. Paths run in parallel until fill or until
    the bid queue empties. b_at_fill = -2 marks a fill by an informed sweep.
    """
    rng = np.random.default_rng(rng)
    A = np.full(n_paths, a); B = np.full(n_paths, b); C = np.full(n_paths, c)
    filled = np.zeros(n_paths, bool)
    bf = np.full(n_paths, -1); cf = np.full(n_paths, -1)
    alive = np.ones(n_paths, bool)
    while alive.any():
        i = np.flatnonzero(alive)
        w = np.column_stack([np.full(i.size, r.lam), r.theta * B[i], r.theta * A[i],
                             np.full(i.size, r.lam), r.mu + r.theta * C[i], np.full(i.size, r.mu),
                             np.full(i.size, r.nu), np.full(i.size, r.nu)])
        cum = np.cumsum(w, axis=1)
        u = rng.random(i.size) * cum[:, -1]
        ev = (u[:, None] > cum).sum(axis=1)
        B[i[ev == 0]] += 1
        B[i[ev == 1]] -= 1
        A[i[ev == 2]] -= 1
        C[i[ev == 3]] += 1
        C[i[ev == 4]] -= 1
        mb = i[ev == 5]
        front = mb[A[mb] == 0]
        A[mb[A[mb] > 0]] -= 1
        filled[front] = True
        bf[front] = B[front]; cf[front] = C[front]
        alive[front] = False
        sweep_up = i[ev == 6]
        filled[sweep_up] = True
        bf[sweep_up] = -2
        alive[sweep_up] = False
        alive[i[ev == 7]] = False
        alive[i[C[i] == 0]] = False
    return filled, bf, cf
