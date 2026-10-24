"""Multivariate Hawkes processes with exponential kernels.

A D-dimensional linear Hawkes process has intensities (1/s)

    lambda_i(t) = mu_i + sum_j sum_{t^j_k < t} alpha_ij exp(-beta_ij (t - t^j_k)),

where mu_i (1/s) is the baseline rate of type i, alpha_ij (1/s) is the jump in
lambda_i caused by an event of type j, and beta_ij (1/s) is the decay rate.
The branching matrix G_ij = alpha_ij / beta_ij is dimensionless; G_ij is the
mean number of type-i events triggered directly by one type-j event. The
process has a stationary version when the spectral radius rho(G) < 1, with
mean intensity vector Lambda = (I - G)^{-1} mu.

Mapping to the paper (Section 6): mu_i is the baseline Lambda^0_u, the
intensity lambda_i(t) is Lambda_u(t), the compensator is A_u(t), and the
observation window T is T_obs. In signature_plot_theory, mu is Lambda^0 per
side, gamma is eta, and tau is the sampling interval h.

Implemented: Ogata thinning simulation, the exact log-likelihood and its
maximization, the time-rescaling residuals, and the closed-form signature
plot of the price model of Bacry, Delattre, Hoffmann and Muzy (2013).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass
class ExpHawkes:
    mu: np.ndarray       # (D,)   baseline intensities (1/s)
    alpha: np.ndarray    # (D, D) jump sizes (1/s)
    beta: np.ndarray     # (D, D) decay rates (1/s)

    def __post_init__(self):
        self.mu = np.atleast_1d(np.asarray(self.mu, dtype=float))
        D = self.mu.size
        self.alpha = np.asarray(self.alpha, dtype=float).reshape(D, D)
        self.beta = np.asarray(self.beta, dtype=float).reshape(D, D)

    @property
    def dim(self) -> int:
        return self.mu.size

    @property
    def branching_matrix(self) -> np.ndarray:
        return self.alpha / self.beta

    @property
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.branching_matrix))))

    def stationary_intensity(self) -> np.ndarray:
        """Lambda = (I - G)^{-1} mu (1/s); requires rho(G) < 1."""
        if self.spectral_radius >= 1:
            raise ValueError("no stationary version: spectral radius of G >= 1")
        return np.linalg.solve(np.eye(self.dim) - self.branching_matrix, self.mu)

    # ------------------------------------------------------------------ simulation
    def simulate(self, T: float, rng=None):
        """Ogata (1981) thinning on [0, T]. Returns (times, types), both sorted.

        With nonnegative alpha the intensity only decays between events, so the
        intensity at the current time bounds it until the next candidate point.
        """
        if np.any(self.alpha < 0):
            raise ValueError("thinning bound requires alpha >= 0")
        rng = np.random.default_rng(rng)
        D = self.dim
        S = np.zeros((D, D))            # S_ij = sum_k exp(-beta_ij (t - t^j_k))
        t = 0.0
        times, types = [], []
        while True:
            lam_bar = float(np.sum(self.mu + np.sum(self.alpha * S, axis=1)))
            w = rng.exponential(1.0 / lam_bar)
            t += w
            if t > T:
                break
            S *= np.exp(-self.beta * w)
            lam = self.mu + np.sum(self.alpha * S, axis=1)
            u = rng.random() * lam_bar
            if u <= lam.sum():
                j = int(np.searchsorted(np.cumsum(lam), u, side="right"))
                j = min(j, D - 1)
                times.append(t)
                types.append(j)
                S[:, j] += 1.0
        return np.asarray(times), np.asarray(types, dtype=np.int64)

    # ------------------------------------------------------------------ inference
    def intensity_at_events(self, times, types):
        """lambda_{type_k}(t_k-) at every event, and the compensators Lambda_i(t_k)."""
        return _recursion(self.mu, self.alpha, self.beta, times, types, T=None)[1:]

    def log_likelihood(self, times, types, T) -> float:
        return _recursion(self.mu, self.alpha, self.beta, times, types, T)[0]

    def time_rescaled_residuals(self, times, types):
        """Compensator increments between consecutive events of each type.

        Under the true model these are i.i.d. Exp(1) (time-rescaling theorem).
        Returns a list of D arrays.
        """
        _, _, comp = _recursion(self.mu, self.alpha, self.beta, times, types, T=None)
        out = []
        for i in range(self.dim):
            c = comp[types == i, i]
            out.append(np.diff(np.concatenate([[0.0], c])))
        return out


def _recursion(mu, alpha, beta, times, types, T):
    """One pass over the events. Returns (loglik, lam_at_events, compensators).

    State S_ij(t) = sum_{t^j_k < t} exp(-beta_ij (t - t^j_k)). Between events
    S decays; at an event of type m, S[:, m] += 1 after the intensity is read.
    The compensator is Lambda_i(t) = mu_i t + sum_j alpha_ij/beta_ij (N_j(t) - S_ij(t)),
    because int_{t_k}^{t} exp(-beta (s - t_k)) ds = (1 - exp(-beta (t - t_k))) / beta.
    """
    D = len(mu)
    mu_l = [float(v) for v in mu]
    a = [[float(alpha[i][j]) for j in range(D)] for i in range(D)]
    b = [[float(beta[i][j]) for j in range(D)] for i in range(D)]
    S = [[0.0] * D for _ in range(D)]
    Ncount = [0] * D
    n = len(times)
    lam_ev = np.empty(n)
    comp = np.empty((n, D))
    ll = 0.0
    t_prev = 0.0
    for k in range(n):
        t = float(times[k])
        m = int(types[k])
        dt = t - t_prev
        for i in range(D):
            Si = S[i]
            bi = b[i]
            for j in range(D):
                Si[j] *= math.exp(-bi[j] * dt)
        lam_m = mu_l[m]
        am = a[m]
        Sm = S[m]
        for j in range(D):
            lam_m += am[j] * Sm[j]
        lam_ev[k] = lam_m
        if lam_m <= 0:
            return -np.inf, lam_ev, comp
        ll += math.log(lam_m)
        for i in range(D):
            c = mu_l[i] * t
            for j in range(D):
                c += a[i][j] / b[i][j] * (Ncount[j] - S[i][j])
            comp[k, i] = c
        for i in range(D):
            S[i][m] += 1.0
        Ncount[m] += 1
        t_prev = t
    if T is not None:
        for i in range(D):
            ll -= mu_l[i] * T
            for j in range(D):
                # remaining decay of S over (t_n, T]
                S_T = S[i][j] * math.exp(-b[i][j] * (T - t_prev))
                ll -= a[i][j] / b[i][j] * (Ncount[j] - S_T)
    return ll, lam_ev, comp


def fit_exp_hawkes(times, types, T, D, x0: ExpHawkes | None = None, rng=None):
    """Maximum-likelihood fit of an exponential-kernel Hawkes process.

    Parameters are optimized on the log scale (all positive). Standard errors
    come from the inverse of the observed information, computed by finite
    differences of the log-likelihood in the natural parameters. They are
    asymptotic (large T) and assume the model is correctly specified.

    Returns (model, se, result) where se is an ExpHawkes holding the standard
    errors in place of the parameters.
    """
    rng = np.random.default_rng(rng)
    if x0 is None:
        rate = np.bincount(types, minlength=D) / T
        x0 = ExpHawkes(mu=0.5 * rate, alpha=np.full((D, D), 0.2 * rate.mean() + 1e-3),
                       beta=np.full((D, D), 1.0 * rate.mean() + 1e-3))

    def unpack(z):
        e = np.exp(z)
        return e[:D], e[D:D + D * D].reshape(D, D), e[D + D * D:].reshape(D, D)

    def nll(z):
        mu, al, be = unpack(z)
        val = _recursion(mu, al, be, times, types, T)[0]
        return -val if np.isfinite(val) else 1e300

    z0 = np.log(np.concatenate([x0.mu, x0.alpha.ravel(), x0.beta.ravel()]))
    res = optimize.minimize(nll, z0, method="L-BFGS-B")
    mu, al, be = unpack(res.x)
    model = ExpHawkes(mu, al, be)

    theta = np.concatenate([mu, al.ravel(), be.ravel()])
    H = _numerical_hessian(lambda th: -_recursion(th[:D], th[D:D + D * D].reshape(D, D),
                                                  th[D + D * D:].reshape(D, D), times, types, T)[0],
                           theta)
    try:
        cov = np.linalg.inv(H)
        se_vec = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se_vec = np.full_like(theta, np.nan)
    se = ExpHawkes(se_vec[:D], se_vec[D:D + D * D], se_vec[D + D * D:])
    return model, se, res


def _numerical_hessian(f, x, rel_step=1e-4):
    n = len(x)
    h = rel_step * np.maximum(np.abs(x), 1e-8)
    H = np.empty((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                xp = x.copy(); xp[i] += h[i]
                xm = x.copy(); xm[i] -= h[i]
                H[i, i] = (f(xp) - 2 * f0 + f(xm)) / h[i] ** 2
            else:
                xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
                xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
                xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
                xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
                H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h[i] * h[j])
    return H


# ---------------------------------------------------------------------- price model
def signature_plot_theory(tau, mu, alpha_s, alpha_c, beta):
    """Realized-variance signature C(tau) = E[(X(t+tau) - X(t))^2] / tau (ticks^2/s).

    Price model of Bacry et al. (2013): X = N_+ - N_-, with N_+ (up ticks) and
    N_- (down ticks) a symmetric bivariate Hawkes process with baseline mu,
    self-excitation alpha_s exp(-beta t) and cross-excitation alpha_c exp(-beta t).
    With g = (alpha_s - alpha_c)/beta, gamma = beta (1 - g) and stationary
    intensity Lambda = mu / (1 - (alpha_s + alpha_c)/beta) per side,

        C(tau) = 2 Lambda [ 1/(1-g)^2 + (1 - 1/(1-g)^2) (1 - exp(-gamma tau)) / (gamma tau) ].

    Limits: C(0+) = 2 Lambda (independent ticks); C(inf) = 2 Lambda / (1-g)^2.
    Cross-excitation (alpha_c > alpha_s) makes g < 0 and C decreasing in tau,
    the mean-reverting microstructure noise seen in tick data.
    Requires (alpha_s + alpha_c) / beta < 1.
    """
    tau = np.asarray(tau, dtype=float)
    if (alpha_s + alpha_c) / beta >= 1:
        raise ValueError("unstable: (alpha_s + alpha_c)/beta must be < 1")
    Lam = mu / (1.0 - (alpha_s + alpha_c) / beta)
    g = (alpha_s - alpha_c) / beta
    gam = beta * (1.0 - g)
    k2 = 1.0 / (1.0 - g) ** 2
    return 2.0 * Lam * (k2 + (1.0 - k2) * (1.0 - np.exp(-gam * tau)) / (gam * tau))


def realized_signature(times, types, T, taus):
    """Empirical C(tau) from one path of X = N_0 - N_1 sampled on a grid of step tau."""
    sign = np.where(types == 0, 1, -1)
    X_ev = np.cumsum(sign)
    out = []
    for tau in taus:
        grid = np.arange(0.0, T, tau)
        idx = np.searchsorted(times, grid, side="right")
        X = np.concatenate([[0], X_ev])[idx]
        dX = np.diff(X)
        out.append(np.mean(dX.astype(float) ** 2) / tau)
    return np.asarray(out)


# ---------------------------------------------------------------------- slowly decaying kernels
@dataclass
class SumExpHawkes1D:
    """Univariate Hawkes process with kernel phi(t) = sum_k w_k beta_k exp(-beta_k t) (1/s).

    Each component integrates to w_k (dimensionless), so the branching ratio is
    n = sum_k w_k. With log-spaced beta_k and weights from
    :func:`power_law_weights`, phi approximates a power law between the
    shortest and the longest time scale. Used to show how an exponential
    kernel fitted on short windows misreads a slowly decaying one.
    """

    mu: float            # baseline intensity (1/s)
    w: np.ndarray        # component weights (dimensionless), n = sum(w)
    beta: np.ndarray     # component decay rates (1/s)

    @property
    def branching_ratio(self) -> float:
        return float(np.sum(self.w))

    def kernel(self, t):
        t = np.asarray(t, float)[..., None]
        return np.sum(self.w * self.beta * np.exp(-self.beta * t), axis=-1)

    def simulate(self, T: float, rng=None) -> np.ndarray:
        """Ogata thinning on [0, T]; returns the sorted event times (s)."""
        rng = np.random.default_rng(rng)
        a = np.asarray(self.w * self.beta, float)
        b = np.asarray(self.beta, float)
        S = np.zeros_like(b)
        t = 0.0
        out = []
        while True:
            lam_bar = self.mu + float(a @ S)
            dt = rng.exponential(1.0 / lam_bar)
            t += dt
            if t > T:
                break
            S *= np.exp(-b * dt)
            if rng.random() * lam_bar <= self.mu + float(a @ S):
                out.append(t)
                S += 1.0
        return np.asarray(out)


def power_law_weights(n, eps, tau0, beta_min, n_comp=16):
    """Weights and rates of a sum of exponentials close to n eps tau0^eps (t + tau0)^-(1+eps).

    Uses (t + tau0)^-(1+eps) = (1/Gamma(1+eps)) int_0^inf s^eps exp(-s (t + tau0)) ds,
    discretized on n_comp log-spaced rates between beta_min and 10/tau0. The
    weights are then rescaled so that sum(w) = n exactly. beta_min (1/s) sets
    the longest memory, 1/beta_min seconds.
    """
    s = np.geomspace(beta_min, 10.0 / tau0, n_comp)
    ds = np.gradient(np.log(s)) * s
    dens = s ** eps * np.exp(-s * tau0) * ds        # weight of exp(-s t) before normalization
    w = dens / s                                     # beta_k exp(-beta_k t) carries 1/beta_k
    return n * w / w.sum(), s


def fit_exp_hawkes_1d(times, T):
    """MLE of mu, alpha, beta for a univariate exponential Hawkes process on [0, T].

    Returns (mu, n_hat = alpha / beta, beta, se_n) with se_n the standard error
    of the branching ratio from the observed information (delta method).
    """
    types = np.zeros(len(times), dtype=np.int64)
    model, se, _ = fit_exp_hawkes(times, types, T, 1)
    mu, al, be = float(model.mu[0]), float(model.alpha[0, 0]), float(model.beta[0, 0])
    # Delta method with the diagonal of the covariance only: a conservative
    # approximation that ignores the alpha-beta covariance.
    se_n = np.sqrt((se.alpha[0, 0] / be) ** 2 + (al * se.beta[0, 0] / be ** 2) ** 2)
    return mu, al / be, be, float(se_n)


def simulate_inhomogeneous_poisson(rate_fn, T, rate_max, rng=None):
    """Poisson process with intensity rate_fn(t) <= rate_max on [0, T], by thinning."""
    rng = np.random.default_rng(rng)
    n = rng.poisson(rate_max * T)
    t = np.sort(rng.uniform(0.0, T, n))
    keep = rng.random(n) * rate_max <= rate_fn(t)
    return t[keep]


def time_change(times, rate_fn, grid):
    """Map event times through the cumulative baseline Lambda(t) = int_0^t rate_fn.

    If rate_fn is the true (or a good estimate of the) seasonal profile, the
    mapped events of an inhomogeneous Poisson process form a unit-rate Poisson
    process. grid is a fine time grid covering the events, used for the integral.
    """
    r = rate_fn(grid)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(grid))])
    return np.interp(times, grid, cum), cum[-1]
