"""Reduced-form Markov model of the best queues (Cont and de Larrard 2013).

Only the two best queues are tracked: q_b orders at the best bid and q_a
orders at the best ask, with a one-tick spread. Until one of them empties,
each queue is an independent birth-death process with constant rates

    up   (limit order joins the queue)          rate lam     (1/s)
    down (market order or cancellation)         rate nu      (1/s)

where nu = mu + theta in the notation of Cont and de Larrard. When the ask
queue empties first the mid price moves up one tick; when the bid queue
empties first it moves down. Both queues are then redrawn from a
regeneration distribution f(x, y), independent of the past.

Main results implemented here:

* the first-passage density of one queue (a continuous-time random walk),
* P(up | q_b, q_a) for general rates by quadrature, and the closed-form
  integral of Cont and de Larrard for the balanced case lam = nu,
* the arctangent law, the continuum limit of P(up) for large queues,
* the survival function of the time to the next price change and its 1/t tail.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, special


def first_passage_density(t, x, lam, nu):
    """Density (1/s) of the time for a queue of size x to reach zero.

    For a continuous-time walk with up rate lam and down rate nu,
        f(t) = (x / t) (nu/lam)^(x/2) I_x(2 sqrt(lam nu) t) exp(-(lam+nu) t),
    with I_x the modified Bessel function. The scaled Bessel function ive is
    used to avoid overflow: I_x(z) e^{-z} = ive(x, z).
    """
    t = np.asarray(t, dtype=float)
    z = 2.0 * np.sqrt(lam * nu) * t
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pref = np.log(x) - np.log(t) + 0.5 * x * np.log(nu / lam)
        f = np.exp(log_pref + z - (lam + nu) * t) * special.ive(x, z)
    return np.where(t > 0, f, 0.0)


def _time_grid(x_max, lam, nu, n=20000):
    """Geometric time grid (s) that resolves first passages of queues up to x_max."""
    scale = max(x_max, 1) ** 2 / (lam + nu)       # diffusive time scale of the queue
    t0 = 1e-4 / (lam + nu)
    return np.concatenate([[0.0], np.geomspace(t0, 1e7 * scale, n)])


def survival(t, x, lam, nu):
    """P(tau_x > t) for one queue of size x.

    Computed by trapezoidal integration of the first-passage density on a fine
    geometric grid, then interpolated at t. When lam > nu the queue may never
    empty and P(tau_x = inf) = 1 - (nu/lam)^x.
    """
    grid = _time_grid(x, lam, nu)
    dens = first_passage_density(grid, x, lam, nu)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    return np.interp(np.asarray(t, dtype=float), grid, 1.0 - cdf)


def p_up(q_b, q_a, lam, nu):
    """P(mid moves up) = P(ask queue empties before the bid queue).

    Computed as  int_0^inf f_{q_a}(t) P(tau_{q_b} > t) dt  on a geometric grid
    that extends to 1e7 diffusive time scales. The truncated tail is below
    P(tau_a > t_max) P(tau_b > t_max), which is under 1e-6 for the default grid.
    Valid for any lam, nu > 0. When lam > nu the price may never move, and the
    result is the probability of an up move in finite time.
    """
    grid = _time_grid(max(q_a, q_b), lam, nu)
    fa = first_passage_density(grid, q_a, lam, nu)
    sb = survival(grid, q_b, lam, nu)
    g = fa * sb
    return float(np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(grid)))


def p_up_balanced(q_b, q_a):
    """Closed form for lam = nu (Cont and de Larrard 2013):

        P(up) = (1/pi) int_0^pi g(u)^{q_a} sin(q_b u) cos(u/2)/sin(u/2) du,
        g(u)  = 2 - cos u - sqrt((2 - cos u)^2 - 1).

    The result does not depend on the common rate. Here q_a is the ask queue,
    whose depletion moves the price up.
    """
    def integrand(u):
        c = 2.0 - np.cos(u)
        g = c - np.sqrt(c * c - 1.0)
        return g ** q_a * np.sin(q_b * u) * np.cos(u / 2) / np.sin(u / 2)

    val, _ = integrate.quad(integrand, 0.0, np.pi, limit=400)
    return val / np.pi


def p_up_arctan(q_b, q_a):
    """Continuum limit for lam = nu and large queues: (2/pi) arctan(q_b / q_a).

    Rescaled, (q_b, q_a) becomes a planar Brownian motion with independent,
    equal-variance coordinates, killed on the axes. The probability of leaving
    through the q_a = 0 axis is harmonic in the quarter plane and equals
    (2/pi) arctan(q_b / q_a).
    """
    return 2.0 / np.pi * np.arctan2(q_b, q_a)


def p_up_linear_system(q_max, lam, nu):
    """P(up) on the grid 1..q_max for both queues, from the embedded jump chain.

    Solves the discrete Dirichlet problem h = P h with h(q_b, 0) = 1 and
    h(0, q_a) = 0, using a reflecting cut-off at M = 4 q_max. Used to test the
    quadrature formulas. Returns an array h[q_b, q_a] of shape (q_max+1, q_max+1).
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    M = 4 * q_max
    n = M * M
    pu = lam / (2 * (lam + nu))
    pd = nu / (2 * (lam + nu))
    A = lil_matrix((n, n))
    rhs = np.zeros(n)

    def ix(b, a):
        return (b - 1) * M + (a - 1)

    for b in range(1, M + 1):
        for a in range(1, M + 1):
            r = ix(b, a)
            A[r, r] = 1.0
            for db, da, p in ((1, 0, pu), (-1, 0, pd), (0, 1, pu), (0, -1, pd)):
                nb, na = min(b + db, M), min(a + da, M)   # reflect at the cut-off
                if nb == 0:
                    continue                               # bid empties: price down, h = 0
                if na == 0:
                    rhs[r] += p                            # ask empties: price up, h = 1
                    continue
                A[r, ix(nb, na)] -= p
    h = spsolve(A.tocsr(), rhs).reshape(M, M)
    out = np.zeros((q_max + 1, q_max + 1))
    out[1:, 1:] = h[:q_max, :q_max]
    return out


def duration_survival(t, q_b, q_a, lam, nu):
    """P(time to next price change > t) = P(tau_{q_b} > t) P(tau_{q_a} > t)."""
    return survival(t, q_b, lam, nu) * survival(t, q_a, lam, nu)


def duration_tail_balanced(t, q_b, q_a, lam):
    """Large-t asymptote of duration_survival for lam = nu:  q_b q_a / (pi lam t).

    Each queue is a symmetric walk with variance rate 2 lam, so
    P(tau_x > t) ~ x / sqrt(pi lam t). The product decays like 1/t, so the
    mean duration is infinite in the balanced case.
    """
    return q_b * q_a / (np.pi * lam * np.asarray(t, dtype=float))


def diffusion_coefficient_balanced(lam, tick, f_mean_product):
    """Price variance rate in the balanced case, sigma^2 = pi lam tick^2 / D(f).

    D(f) = sum_{x,y} x y f(x, y) is the mean product of the regenerated queue
    sizes. Cont and de Larrard (2013) prove that s(t n log n)/sqrt(n) converges
    to sigma W, with W a standard Brownian motion. Units: price^2 / s.
    """
    return np.pi * lam * tick ** 2 / f_mean_product


def simulate_durations(q_b, q_a, lam, nu, n_paths, t_max, rng=None):
    """Monte Carlo of the time to the next price change and its direction.

    Simulates n_paths copies of the two-queue walk in parallel until one queue
    empties or t_max is reached. Returns (tau, up) where tau = inf for paths
    still alive at t_max and up is True when the ask queue emptied first.
    """
    rng = np.random.default_rng(rng)
    b = np.full(n_paths, q_b, dtype=np.int64)
    a = np.full(n_paths, q_a, dtype=np.int64)
    t = np.zeros(n_paths)
    tau = np.full(n_paths, np.inf)
    up = np.zeros(n_paths, dtype=bool)
    alive = np.ones(n_paths, dtype=bool)
    rate = 2.0 * (lam + nu)                       # total jump rate of the pair
    p_up_jump = lam / (lam + nu)
    while alive.any():
        idx = np.flatnonzero(alive)
        t[idx] += rng.exponential(1.0 / rate, idx.size)
        over = t[idx] > t_max
        alive[idx[over]] = False
        idx = idx[~over]
        which = rng.random(idx.size) < 0.5         # True: bid queue moves
        step = np.where(rng.random(idx.size) < p_up_jump, 1, -1)
        b[idx[which]] += step[which]
        a[idx[~which]] += step[~which]
        done_b = b[idx] == 0
        done_a = a[idx] == 0
        tau[idx[done_b | done_a]] = t[idx[done_b | done_a]]
        up[idx[done_a]] = True
        alive[idx[done_b | done_a]] = False
    return tau, up
