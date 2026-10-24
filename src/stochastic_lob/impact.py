"""Price impact: the propagator model and order flow imbalance.

Propagator model (Bouchaud, Gefen, Potters and Wyart 2004). Time is counted in
trades (event time n). Trade signs eps_n = +1 (buy) or -1 (sell) have long
memory, C(l) = E[eps_n eps_{n+l}] ~ c0 l^(-gamma_eps) with 0 < gamma_eps < 1.
The price just before trade n, in ticks, is

    P_n = sum_{m < n} G(n - m) eps_m,      G(l) = Gamma0 l^(-zeta),  l >= 1,

with Gamma0 the impact one trade later (ticks) and zeta the decay exponent.
For large lags the price variance grows like l^(2 - gamma_eps - 2 zeta), so
prices diffuse (variance linear in l) only when zeta = zeta_c = (1 - gamma_eps)/2.
Faster decay gives mean reversion, slower decay gives trending prices.

Code names match the paper: gamma_eps is the sign-correlation exponent and
zeta the kernel exponent.

Order flow imbalance (Cont, Kukanov and Stoikov 2014). From consecutive
best-quote snapshots (P^B, q^B, P^A, q^A) the OFI contribution of snapshot n is

    e_n = 1{P^B_n >= P^B_{n-1}} q^B_n - 1{P^B_n <= P^B_{n-1}} q^B_{n-1}
        - 1{P^A_n <= P^A_{n-1}} q^A_n + 1{P^A_n >= P^A_{n-1}} q^A_{n-1},

and OFI over an interval is the sum of e_n inside it.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------- signs
def fgn_autocovariance(n, H):
    """Autocovariance of unit-variance fractional Gaussian noise at lags 0..n-1."""
    k = np.arange(n, dtype=float)
    return 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))


def fgn(n, H, rng=None):
    """Exact fractional Gaussian noise of length n by circulant embedding (Davies-Harte)."""
    rng = np.random.default_rng(rng)
    g = fgn_autocovariance(n + 1, H)
    c = np.concatenate([g, g[-2:0:-1]])           # first row of the 2n circulant
    lam = np.fft.fft(c).real
    if np.any(lam < -1e-10):
        raise ValueError("circulant embedding not nonnegative definite")
    lam = np.clip(lam, 0, None)
    m = len(c)
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    return np.fft.fft(np.sqrt(lam / m) * z).real[:n]


def long_memory_signs(n, gamma_eps, rng=None):
    """Trade signs eps = sign(fGn) with Hurst H = 1 - gamma_eps/2.

    For a Gaussian pair with correlation r, E[sign X sign Y] = (2/pi) arcsin r.
    Since the fGn correlation decays as H(2H-1) l^(2H-2), the sign correlation
    decays as (2/pi) H (2H-1) l^(-gamma_eps) for large l.
    """
    if not 0 < gamma_eps < 1:
        raise ValueError("long memory requires 0 < gamma_eps < 1")
    H = 1.0 - gamma_eps / 2.0
    return np.sign(fgn(n, H, rng)).astype(float)


def sign_autocorrelation_theory(lags, gamma_eps):
    """(2/pi) arcsin(rho_H(l)), the exact sign correlation of sign(fGn)."""
    H = 1.0 - gamma_eps / 2.0
    r = fgn_autocovariance(int(np.max(lags)) + 1, H)[np.asarray(lags)]
    return 2.0 / np.pi * np.arcsin(r)


def autocorrelation(x, lags):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / np.arange(n, 0, -1)
    return acf[np.asarray(lags)] / acf[0]


# ---------------------------------------------------------------------- propagator
def propagator_kernel(lags, Gamma0=1.0, zeta=0.25):
    """G(l) = Gamma0 l^(-zeta) for l >= 1 (ticks)."""
    return Gamma0 * np.asarray(lags, float) ** (-zeta)


def propagator_price(signs, Gamma0=1.0, zeta=0.25):
    """P_n = sum_{m<n} G(n-m) eps_m, the price before trade n, by FFT convolution."""
    n = len(signs)
    G = np.zeros(n)
    G[1:] = propagator_kernel(np.arange(1, n), Gamma0, zeta)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    return np.fft.irfft(np.fft.rfft(signs, m) * np.fft.rfft(G, m), m)[:n]


def response_function(prices, signs, lags):
    """R(l) = E[eps_n (P_{n+l} - P_n)] (ticks), with P_n the price before trade n."""
    out = []
    for l in lags:
        out.append(np.mean(signs[:-l] * (prices[l:] - prices[:-l])))
    return np.asarray(out)


def diffusion_signature(prices, lags):
    """D(l) = Var(P_{n+l} - P_n) / l (ticks^2 per trade)."""
    out = []
    for l in lags:
        d = prices[l:] - prices[:-l]
        out.append(np.var(d) / l)
    return np.asarray(out)


def critical_zeta(gamma_eps):
    """Kernel exponent zeta_c = (1 - gamma_eps)/2 that makes prices diffusive."""
    return 0.5 * (1.0 - gamma_eps)


# ---------------------------------------------------------------------- OFI
def ofi_contributions(bid_p, bid_q, ask_p, ask_q):
    """Per-snapshot OFI e_n (orders) for n = 1..N-1, from best-quote arrays."""
    bp0, bp1 = bid_p[:-1], bid_p[1:]
    ap0, ap1 = ask_p[:-1], ask_p[1:]
    bq0, bq1 = bid_q[:-1], bid_q[1:]
    aq0, aq1 = ask_q[:-1], ask_q[1:]
    return ((bp1 >= bp0) * bq1 - (bp1 <= bp0) * bq0
            - (ap1 <= ap0) * aq1 + (ap1 >= ap0) * aq0).astype(float)


def aggregate_ofi(t, bid_p, bid_q, ask_p, ask_q, dt):
    """OFI and mid-price change (ticks) over consecutive windows of length dt (s)."""
    e = ofi_contributions(bid_p, bid_q, ask_p, ask_q)
    mid = 0.5 * (bid_p + ask_p)
    edges = np.arange(t[0], t[-1], dt)
    win = np.searchsorted(edges, t[1:], side="right") - 1
    n_win = len(edges) - 1
    keep = (win >= 0) & (win < n_win)
    ofi = np.bincount(win[keep], weights=e[keep], minlength=n_win)
    idx = np.searchsorted(t, edges, side="right") - 1
    dmid = np.diff(mid[np.clip(idx, 0, len(mid) - 1)])
    return ofi, dmid


def ols(x, y):
    """y = a + b x by least squares. Returns dict with estimates, standard errors, R^2, n.

    Standard errors are the classical homoskedastic ones; they assume
    uncorrelated residuals across windows.
    """
    res = stats.linregress(x, y)
    return {"slope": res.slope, "slope_se": res.stderr, "intercept": res.intercept,
            "intercept_se": res.intercept_stderr, "r2": res.rvalue ** 2, "n": len(x)}


# ---------------------------------------------------------------------- multi-level OFI
def multilevel_ofi_contributions(bid_p, bid_depth, ask_p, ask_depth):
    """Per-snapshot OFI at levels m = 0..K-1 (Xu, Gould and Howison 2019).

    Level m on the bid side is the price p_B - m ticks and on the ask side
    p_A + m ticks. The CKS formula is applied level by level, so column 0 is
    the best-level OFI. Returns an (N-1, K) array in orders.
    """
    K = bid_depth.shape[1]
    out = np.empty((len(bid_p) - 1, K))
    for m in range(K):
        out[:, m] = ofi_contributions(bid_p - m, bid_depth[:, m], ask_p + m, ask_depth[:, m])
    return out


def aggregate_multilevel_ofi(t, bid_p, bid_depth, ask_p, ask_depth, dt):
    """Multi-level OFI (n_win, K) and mid change (ticks) over windows of length dt (s)."""
    e = multilevel_ofi_contributions(bid_p, bid_depth, ask_p, ask_depth)
    mid = 0.5 * (bid_p + ask_p)
    edges = np.arange(t[0], t[-1], dt)
    win = np.searchsorted(edges, t[1:], side="right") - 1
    n_win = len(edges) - 1
    keep = (win >= 0) & (win < n_win)
    ofi = np.column_stack([np.bincount(win[keep], weights=e[keep, m], minlength=n_win)
                           for m in range(e.shape[1])])
    idx = np.searchsorted(t, edges, side="right") - 1
    dmid = np.diff(mid[np.clip(idx, 0, len(mid) - 1)])
    return ofi, dmid


def integrated_ofi(ofi_levels, mean_depth):
    """Integrated OFI of Cont, Cucuringu and Zhang (2023).

    Each level's OFI is divided by the mean depth of the first K levels, and
    the series are combined with the weights of the first principal component,
    normalized to unit L1 norm. Returns (weights, integrated series).
    """
    X = ofi_levels / mean_depth
    C = np.cov(X, rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    w = vecs[:, -1]
    w = w * np.sign(w.sum())
    w = w / np.abs(w).sum()
    return w, X @ w


def ols_multi(X, y):
    """y = a + X b by least squares, with classical standard errors and R^2."""
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    n, p = X1.shape
    s2 = resid @ resid / (n - p)
    cov = s2 * np.linalg.inv(X1.T @ X1)
    r2 = 1.0 - resid @ resid / np.sum((y - y.mean()) ** 2)
    return {"coef": beta, "se": np.sqrt(np.diag(cov)), "r2": r2, "n": n}


def out_of_sample_r2(X, y, train_frac=0.5):
    """Fit on the first train_frac of windows, report R^2 on the rest."""
    n_tr = int(train_frac * len(y))
    fit = ols_multi(X[:n_tr], y[:n_tr])
    X1 = np.column_stack([np.ones(len(y) - n_tr), X[n_tr:]])
    resid = y[n_tr:] - X1 @ fit["coef"]
    return 1.0 - resid @ resid / np.sum((y[n_tr:] - y[n_tr:].mean()) ** 2)


# ---------------------------------------------------------------------- propagator estimation
def increment_kernel(G):
    """Increments g_0 = G(1), g_k = G(k+1) - G(k) of a propagator given as G[l-1] = G(l)."""
    G = np.asarray(G, float)
    return np.concatenate([[G[0]], np.diff(G)])


def estimate_propagator(signs, dprice, max_lag):
    """Estimate G(l), l = 1..max_lag, from trade signs and price changes (ticks).

    With P_n = sum_{m<n} G(n-m) eps_m, the change dP_n = P_{n+1} - P_n equals
    sum_{k>=0} g_k eps_{n-k} with g_0 = G(1) and g_k = G(k+1) - G(k). Any
    noise in dP_n that is uncorrelated with past signs drops out. Multiplying
    by eps_{n-j} and averaging gives the Toeplitz system

        sum_k C(|j - k|) g_k = S(j),   C(l) = E[eps_n eps_{n+l}],  S(j) = E[dP_n eps_{n-j}],

    solved for g_0..g_{max_lag-1} (Bouchaud et al. 2004; Eisler, Bouchaud and
    Kockelkoren 2012). Then G(l) = sum_{k<l} g_k.
    """
    from scipy.linalg import solve_toeplitz

    eps = np.asarray(signs, float)
    dp = np.asarray(dprice, float)
    n = len(dp)
    eps = eps[:n]
    m = 1 << int(np.ceil(np.log2(2 * n)))
    fe = np.fft.rfft(eps, m)
    C = np.fft.irfft(fe * np.conj(fe), m)[:max_lag] / (n - np.arange(max_lag))
    fd = np.fft.rfft(dp, m)
    # S(j) = mean_n dp[n] eps[n - j]  (cross-correlation at lag j)
    S = np.fft.irfft(fd * np.conj(fe), m)[:max_lag] / (n - np.arange(max_lag))
    g = solve_toeplitz(C, S)
    return np.cumsum(g)


def block_bootstrap(stat, n, n_blocks=20, n_boot=200, rng=None):
    """Standard error of stat(index_array) by resampling contiguous blocks.

    stat receives an array of indices (a concatenation of resampled blocks).
    Blocks must be long compared with the memory of the data.
    """
    rng = np.random.default_rng(rng)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_blocks)]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_blocks, n_blocks)
        vals.append(stat(np.concatenate([blocks[i] for i in pick])))
    return np.std(np.asarray(vals), axis=0, ddof=1)
