"""Optimal market making in a limit order book.

Setting of Avellaneda and Stoikov (2008). The mid price follows
S_t = S_0 + sigma W_t, with sigma the volatility (price / sqrt(s)). A market
maker quotes a bid at S - delta_b and an ask at S + delta_a. Each quote is hit
by a Poisson stream with intensity A exp(-k delta), where A (1/s) is the rate
at zero distance and k (1/price) sets how fast fills thin out. The maker holds
q (shares, signed) and cash X, and maximizes E[-exp(-gamma (X_T + q_T S_T))]
with risk aversion gamma (1/price, for unit share size) and horizon T (s).

Avellaneda-Stoikov approximation (from an expansion in the inventory):

    reservation price   r = S - q gamma sigma^2 (T - t)
    total spread        delta_a + delta_b = gamma sigma^2 (T - t) + (2/gamma) ln(1 + gamma/k)

with quotes placed symmetrically around r.

Gueant, Lehalle and Fernandez-Tapia (2013) solve the problem with inventory
limits and give the large-T (stationary) quotes

    delta_b(q) = (1/gamma) ln(1 + gamma/k) + (2q + 1)/2 * omega,
    delta_a(q) = (1/gamma) ln(1 + gamma/k) - (2q - 1)/2 * omega,
    omega      = sqrt( sigma^2 gamma / (2 k A) * (1 + gamma/k)^(1 + k/gamma) ),

where omega has units of price.

Exact solution with inventory limits (GLFT 2013, Proposition 2 and Theorem 1).
With |q| <= Q, let alpha = (k/2) gamma sigma^2 and eta = A (1 + gamma/k)^-(1 + k/gamma).
With a price drift m (price/s), let b = k m. The tridiagonal matrix M has
diagonal alpha q^2 - b q and off-diagonals -eta, and

    v(t) = exp(-M (T - t)) 1,
    delta_b(t, q) = (1/k) ln(v_q / v_{q+1}) + (1/gamma) ln(1 + gamma/k),   q < Q,
    delta_a(t, q) = (1/k) ln(v_q / v_{q-1}) + (1/gamma) ln(1 + gamma/k),   q > -Q.

The drift case is GLFT Proposition 4. As T - t grows, v aligns with the
eigenvector of M for its smallest eigenvalue, which gives the stationary
quotes (GLFT Theorem 2).

Mapping to the slides: q is the inventory y, X the cash x, delta_b and delta_a
the quote distances d^b and d^a, and drift is the short-term alpha m.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MMParams:
    S0: float = 100.0     # initial mid price (price)
    T: float = 1.0        # horizon (s)
    sigma: float = 2.0    # volatility (price / sqrt(s))
    dt: float = 0.005     # time step (s)
    gamma: float = 0.1    # risk aversion (1 / price)
    k: float = 1.5        # fill-intensity decay (1 / price)
    A: float = 140.0      # fill intensity at zero distance (1 / s)
    drift: float = 0.0    # mid-price drift m (price / s)


def as_quotes(S, q, t, p: MMParams):
    """Avellaneda-Stoikov bid and ask distances (price) from the mid S."""
    tau = p.T - t
    r = S - q * p.gamma * p.sigma ** 2 * tau
    half = 0.5 * (p.gamma * p.sigma ** 2 * tau + 2.0 / p.gamma * np.log1p(p.gamma / p.k))
    bid = r - half
    ask = r + half
    return S - bid, ask - S


def glft_quotes(q, p: MMParams):
    """Gueant-Lehalle-Fernandez-Tapia asymptotic bid and ask distances (price)."""
    base = np.log1p(p.gamma / p.k) / p.gamma
    omega = np.sqrt(p.sigma ** 2 * p.gamma / (2 * p.k * p.A) * (1 + p.gamma / p.k) ** (1 + p.k / p.gamma))
    return base + (2 * q + 1) / 2 * omega, base - (2 * q - 1) / 2 * omega


def symmetric_quotes(S, q, t, p: MMParams):
    """Benchmark: the AS spread at time t, centred on the mid instead of r."""
    tau = p.T - t
    half = 0.5 * (p.gamma * p.sigma ** 2 * tau + 2.0 / p.gamma * np.log1p(p.gamma / p.k))
    return np.full_like(np.asarray(q, float), half), np.full_like(np.asarray(q, float), half)


def simulate(strategy, p: MMParams, n_paths=1000, rng=None):
    """Simulate the market maker on n_paths independent paths.

    strategy(S, q, t, p) returns (delta_b, delta_a) arrays. The mid moves by
    +/- sigma sqrt(dt) each step, as in AS (2008). In each step a quote at
    distance delta is filled with probability A exp(-k delta) dt, which must be
    small (A dt = 0.7 for the default parameters, so the per-step fill
    probability at the typical distance is below 0.3).

    Returns dict with terminal profit X_T + q_T S_T - (X_0 + q_0 S_0) and
    terminal inventory, one entry per path, plus an inventory path sample.
    """
    rng = np.random.default_rng(rng)
    n_steps = int(round(p.T / p.dt))
    S = np.full(n_paths, p.S0)
    q = np.zeros(n_paths)
    X = np.zeros(n_paths)
    q_path = np.empty((n_steps + 1, min(n_paths, 5)))
    q_path[0] = q[: q_path.shape[1]]
    for n in range(n_steps):
        t = n * p.dt
        db, da = strategy(S, q, t, p)
        pb = np.clip(p.A * np.exp(-p.k * db) * p.dt, 0, 1)
        pa = np.clip(p.A * np.exp(-p.k * da) * p.dt, 0, 1)
        buy = rng.random(n_paths) < pb       # our bid is hit: we buy one share
        sell = rng.random(n_paths) < pa      # our ask is lifted: we sell one share
        X += np.where(sell, S + da, 0.0) - np.where(buy, S - db, 0.0)
        q += buy.astype(float) - sell.astype(float)
        S += p.drift * p.dt + p.sigma * np.sqrt(p.dt) * np.where(rng.random(n_paths) < 0.5, 1.0, -1.0)
        q_path[n + 1] = q[: q_path.shape[1]]
    return {"profit": X + q * S, "q_T": q, "q_path": q_path}


# ---------------------------------------------------------------------- exact GLFT
def glft_matrix(Q: int, p: MMParams, drift: float | None = None) -> np.ndarray:
    """The (2Q+1) x (2Q+1) matrix M of GLFT, rows ordered q = -Q..Q."""
    m = p.drift if drift is None else drift
    alpha = 0.5 * p.k * p.gamma * p.sigma ** 2
    eta = p.A * (1 + p.gamma / p.k) ** (-(1 + p.k / p.gamma))
    q = np.arange(-Q, Q + 1)
    M = np.diag(alpha * q ** 2 - p.k * m * q)
    M -= eta * (np.eye(2 * Q + 1, k=1) + np.eye(2 * Q + 1, k=-1))
    return M


def glft_value_vectors(Q: int, p: MMParams, taus, drift: float | None = None) -> np.ndarray:
    """v at time-to-horizon tau = T - t, rows q = -Q..Q, one column per tau.

    M is symmetric, so v = U exp(-L tau) U^T 1. Each column is scaled by
    exp(lambda_min tau) to avoid underflow; quotes depend only on ratios of v.
    """
    M = glft_matrix(Q, p, drift)
    L, U = np.linalg.eigh(M)
    c = U.T @ np.ones(2 * Q + 1)
    taus = np.atleast_1d(np.asarray(taus, float))
    return U @ (c[:, None] * np.exp(-(L - L[0])[:, None] * taus[None, :]))


def glft_exact_quotes(Q: int, p: MMParams, taus, drift: float | None = None):
    """Exact optimal bid and ask distances, arrays of shape (len(taus), 2Q+1).

    Entry [j, q + Q] is the quote at time-to-horizon taus[j] and inventory q.
    The bid is +inf at q = Q and the ask is +inf at q = -Q (no quote).
    """
    v = glft_value_vectors(Q, p, taus, drift).T
    base = np.log1p(p.gamma / p.k) / p.gamma
    db = np.full(v.shape, np.inf)
    da = np.full(v.shape, np.inf)
    db[:, :-1] = np.log(v[:, :-1] / v[:, 1:]) / p.k + base
    da[:, 1:] = np.log(v[:, 1:] / v[:, :-1]) / p.k + base
    return db, da


def glft_stationary_quotes(Q: int, p: MMParams, drift: float | None = None):
    """Long-horizon quotes from the ground-state eigenvector of M (GLFT Theorem 2)."""
    M = glft_matrix(Q, p, drift)
    L, U = np.linalg.eigh(M)
    f = np.abs(U[:, 0])
    base = np.log1p(p.gamma / p.k) / p.gamma
    db = np.full(2 * Q + 1, np.inf)
    da = np.full(2 * Q + 1, np.inf)
    db[:-1] = np.log(f[:-1] / f[1:]) / p.k + base
    da[1:] = np.log(f[1:] / f[:-1]) / p.k + base
    return db, da


def glft_strategy(Q: int, p: MMParams, drift: float | None = None):
    """Strategy callable for simulate(): exact GLFT quotes looked up on the time grid."""
    n_steps = int(round(p.T / p.dt))
    taus = p.T - np.arange(n_steps) * p.dt
    db, da = glft_exact_quotes(Q, p, taus, drift)

    def strategy(S, q, t, pp):
        j = min(int(round(t / pp.dt)), n_steps - 1)
        qi = np.clip(np.rint(q).astype(int), -Q, Q) + Q
        return db[j, qi], da[j, qi]

    return strategy


def certainty_equivalent(profit, gamma):
    """CARA certainty equivalent -(1/gamma) ln E[exp(-gamma Pi)] (price units)."""
    z = -gamma * np.asarray(profit)
    zmax = z.max()
    return -(zmax + np.log(np.mean(np.exp(z - zmax)))) / gamma


def simulate_regime_signal(p: MMParams, Q: int, m0: float, switch_rate: float,
                           use_signal: bool, n_paths=2000, rng=None, shrink: float = 1.0):
    """Market making when the mid drifts at +m0 or -m0, switching at switch_rate (1/s).

    The maker sees the current regime. With use_signal the maker quotes the
    exact GLFT quotes for drift shrink * (current drift), a heuristic that
    ignores future switches; otherwise the drift-free GLFT quotes. GLFT treats
    the drift as permanent, so a short-lived signal should be shrunk.
    Returns profit and terminal inventory per path.
    """
    rng = np.random.default_rng(rng)
    n_steps = int(round(p.T / p.dt))
    taus = p.T - np.arange(n_steps) * p.dt
    ms = shrink * m0
    tables = {m: glft_exact_quotes(Q, p, taus, m) for m in (ms, -ms, 0.0)}
    S = np.full(n_paths, p.S0)
    q = np.zeros(n_paths)
    X = np.zeros(n_paths)
    regime = np.where(rng.random(n_paths) < 0.5, 1.0, -1.0)
    for n in range(n_steps):
        qi = (np.clip(q, -Q, Q) + Q).astype(int)
        if use_signal:
            db = np.where(regime > 0, tables[ms][0][n, qi], tables[-ms][0][n, qi])
            da = np.where(regime > 0, tables[ms][1][n, qi], tables[-ms][1][n, qi])
        else:
            db, da = tables[0.0][0][n, qi], tables[0.0][1][n, qi]
        buy = rng.random(n_paths) < np.clip(p.A * np.exp(-p.k * db) * p.dt, 0, 1)
        sell = rng.random(n_paths) < np.clip(p.A * np.exp(-p.k * da) * p.dt, 0, 1)
        X += np.where(sell, S + da, 0.0) - np.where(buy, S - db, 0.0)
        q += buy.astype(float) - sell.astype(float)
        S += regime * m0 * p.dt + p.sigma * np.sqrt(p.dt) * rng.standard_normal(n_paths)
        flip = rng.random(n_paths) < switch_rate * p.dt
        regime[flip] *= -1.0
    return {"profit": X + q * S, "q_T": q}


# ---------------------------------------------------------------------- fill-intensity estimation
def simulate_quote_spells(A, k, distances, t_max, rng=None):
    """Quote spells for estimating the fill intensity A exp(-k d).

    Each spell posts one quote at distance d (price) and keeps it until it is
    filled or until t_max (s), whichever comes first; then it is cancelled.
    Returns (exposure, filled): the time the quote rested (s) and whether the
    spell ended in a fill. Spells that end at t_max are right-censored.
    """
    rng = np.random.default_rng(rng)
    d = np.asarray(distances, float)
    wait = rng.exponential(1.0 / (A * np.exp(-k * d)))
    filled = wait < t_max
    return np.minimum(wait, t_max), filled


def fit_fill_intensity(d, exposure, filled, iters=50):
    """Poisson-exposure MLE of (A, k) in the fill intensity A exp(-k d).

    The log-likelihood of spells with exposures tau_i and fill indicators n_i is
    sum_i [n_i (ln A - k d_i) - A exp(-k d_i) tau_i], which treats censored
    spells correctly. Newton's method on (ln A, k). Returns (A, k, cov) with
    cov the inverse Fisher information for (ln A, k).
    """
    d = np.asarray(d, float)
    tau = np.asarray(exposure, float)
    n = np.asarray(filled, float)
    X = np.column_stack([np.ones_like(d), -d])
    b = np.array([np.log(max(n.sum(), 1.0) / tau.sum()), 0.0])
    for _ in range(iters):
        mu = np.exp(X @ b) * tau
        grad = X.T @ (n - mu)
        H = X.T @ (X * mu[:, None])
        step = np.linalg.solve(H, grad)
        b += step
        if np.max(np.abs(step)) < 1e-12:
            break
    mu = np.exp(X @ b) * tau
    cov = np.linalg.inv(X.T @ (X * mu[:, None]))
    return float(np.exp(b[0])), float(b[1]), cov


def fit_fill_ratio_naive(d, filled, t_max):
    """Naive estimate: regress ln(fill fraction / t_max) on d, per distinct d.

    Ignores that a quote can fill at most once in t_max, so the fill fraction
    1 - exp(-A e^{-kd} t_max) saturates near the touch and k is biased down.
    Returns (A, k).
    """
    d = np.asarray(d, float)
    ds = np.unique(d)
    frac = np.array([np.mean(filled[d == x]) for x in ds])
    ok = (frac > 0) & (frac < 1)
    slope, icpt = np.polyfit(ds[ok], np.log(frac[ok] / t_max), 1)
    return float(np.exp(icpt)), float(-slope)
