"""Latent order book and metaorder impact (after Donier, Bonart, Mastromatteo and Bouchaud 2015).

The latent book holds the buy and sell intentions that are not shown in the
visible book. Write b(x, t) and a(x, t) for the latent bid and ask densities
(shares per tick) at price x (ticks), and phi = b - a. In the simplest version
(no deposition or cancellation, infinite memory) intentions diffuse with
coefficient D (ticks^2 / s) and annihilate when they meet, so

    d phi / dt = D d^2 phi / dx^2 + m(t) delta(x - y_t),

where y_t, the price, is the zero of phi. A buy metaorder executed at rate
m(t) (shares / s) removes ask liquidity at the price, which adds +m to phi.
Far from the price the book is linear: phi(x, 0) = -L x, with L (shares /
tick^2) the slope of the latent book.

Dimensions. The only dimensionless group made from m, L and D is

    u = m / (L D),

the execution rate relative to the rate at which diffusion refills the
latent book near the price. For constant m the problem has no time scale, so
the price path is self-similar: y_t = sqrt(D t) F(u).

Closed form for F (derived for this repository; see the topic slides).
Write phi as the free solution -L x plus the heat kernel response to the
source, set phi(y_t, t) = 0 with y_s = A sqrt(s), and substitute s = t w and
w = 1 - v^2. With a = A / sqrt(D):

    a = (u / sqrt(pi)) * int_0^1 exp(-a^2 r(v) / 4) dv,
    r(v) = (1 - sqrt(1 - v^2)) / (1 + sqrt(1 - v^2)).

The right side decreases in a and the left side increases, so the root is
unique. Limits: a -> u / sqrt(pi) for small u (linear impact) and
a -> sqrt(2 u) for large u, which gives I = sqrt(2 Q / L) for a metaorder of
Q = m T shares (square-root impact).

Linear-response regime (small u), for a metaorder that runs on [0, T]:

    y_t = (m / (L sqrt(pi D))) * (sqrt(t) - sqrt(max(t - T, 0))),

so impact decays after the metaorder ends like a propagator with exponent 1/2.

The finite-difference solver below checks both results.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, optimize
from scipy.linalg import solve_banded


def _r(v):
    s = np.sqrt(1.0 - v * v)
    return (1.0 - s) / (1.0 + s)


def self_similar_amplitude(u: float) -> float:
    """F(u) = y_t / sqrt(D t) for constant execution rate, u = m / (L D)."""
    if u <= 0:
        return 0.0

    def g(a):
        val, _ = integrate.quad(lambda v: np.exp(-a * a * _r(v) / 4.0), 0.0, 1.0, limit=200)
        return a - u / np.sqrt(np.pi) * val

    hi = max(2.0 * np.sqrt(2.0 * u), 2.0 * u / np.sqrt(np.pi)) + 1.0
    return float(optimize.brentq(g, 1e-14, hi, xtol=1e-13))


def impact_peak(Q, T, L, D):
    """Peak impact (ticks) of a metaorder of Q shares executed uniformly over T seconds."""
    u = Q / (T * L * D)
    return np.sqrt(D * T) * self_similar_amplitude(u)


def linear_response_path(t, m, T, L, D):
    """Small-u price path y_t (ticks) during and after a metaorder on [0, T]."""
    t = np.asarray(t, float)
    return m / (L * np.sqrt(np.pi * D)) * (np.sqrt(np.clip(t, 0, None)) - np.sqrt(np.clip(t - T, 0, None)))


def simulate_metaorder(m, T, L=1.0, D=1.0, t_end=None, n_x=8001, n_t=4000, x_max=None):
    """Finite-difference solution of the latent book with a buy metaorder at rate m on [0, T].

    Crank-Nicolson in time on a uniform price grid, with backward Euler for the
    first four steps to damp the start-up oscillation of the kink at the
    price. Time steps grow as t_n = T (n / n_t)^2 during execution, because
    the price moves like sqrt(t), and are uniform after it. The source is
    spread over the two grid nodes around the current price with linear
    weights. Dirichlet boundaries hold phi = -L x at +-x_max. Returns (t, y)
    with y the price (ticks) at each time step.

    By default x_max covers the square-root impact plus eight diffusion lengths.
    """
    t_end = T if t_end is None else t_end
    if x_max is None:
        x_max = 1.5 * np.sqrt(2 * m * T / L) + 8.0 * np.sqrt(D * t_end) + 1.0
    x = np.linspace(-x_max, x_max, n_x)
    h = x[1] - x[0]
    t_exec = T * (np.arange(n_t + 1) / n_t) ** 2
    n_after = int(np.ceil(n_t * max(t_end - T, 0.0) / T))
    t_post = T + (t_end - T) * np.arange(1, n_after + 1) / max(n_after, 1) if n_after else np.empty(0)
    ts = np.concatenate([t_exec, t_post])
    phi = -L * x
    left, right = phi[0], phi[-1]
    y = 0.0
    ys = np.zeros(len(ts))
    for n in range(len(ts) - 1):
        dt = ts[n + 1] - ts[n]
        r = D * dt / h ** 2
        theta = 1.0 if n < 4 else 0.5          # implicit weight
        ab = np.zeros((3, n_x))
        ab[0, 2:] = -theta * r
        ab[1, :] = 1 + 2 * theta * r
        ab[2, :-2] = -theta * r
        ab[1, 0] = ab[1, -1] = 1.0
        rhs = phi.copy()
        rhs[1:-1] = phi[1:-1] + (1 - theta) * r * (phi[2:] - 2 * phi[1:-1] + phi[:-2])
        if ts[n] < T:
            mass = m * (min(ts[n + 1], T) - ts[n])
            j = int(np.clip(np.floor((y - x[0]) / h), 0, n_x - 2))
            w = (y - x[j]) / h
            rhs[j] += mass * (1 - w) / h
            rhs[j + 1] += mass * w / h
        rhs[0], rhs[-1] = left, right
        phi = solve_banded((1, 1), ab, rhs)
        y = _zero(x, phi, y)
        ys[n + 1] = y
    return ts, ys


def _zero(x, phi, guess):
    """Price: the zero crossing of phi (positive on the left) nearest the previous price."""
    s = np.flatnonzero((phi[:-1] > 0) & (phi[1:] <= 0))
    if s.size == 0:
        raise RuntimeError("no zero crossing; enlarge the grid")
    j = s[np.argmin(np.abs(x[s] - guess))]
    return float(x[j] + phi[j] / (phi[j] - phi[j + 1]) * (x[j + 1] - x[j]))


def local_exponent(Q, T, L, D):
    """d ln I / d ln Q along the self-similar curve (1 = linear, 1/2 = square root)."""
    Q = np.asarray(Q, float)
    eps = 1e-3
    lo = np.array([impact_peak(q * (1 - eps), T, L, D) for q in Q])
    hi = np.array([impact_peak(q * (1 + eps), T, L, D) for q in Q])
    return (np.log(hi) - np.log(lo)) / (np.log(1 + eps) - np.log(1 - eps))
