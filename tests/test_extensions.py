"""Checks of the estimation and deep-dive modules against results known independently.

Run with  pytest  or, without pytest,  python tests/test_extensions.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stochastic_lob import adverse_selection as gm  # noqa: E402
from stochastic_lob import hawkes as hk  # noqa: E402
from stochastic_lob import impact as im  # noqa: E402
from stochastic_lob import latent_book as lb  # noqa: E402
from stochastic_lob import lobster, microprice as mp  # noqa: E402
from stochastic_lob import market_making as mm  # noqa: E402
from stochastic_lob import markov_lob as ml  # noqa: E402
from stochastic_lob import queue_reactive as qr  # noqa: E402


# ---------------------------------------------------------------- latent order book
def test_self_similar_limits():
    # small u: linear impact u / sqrt(pi); large u: square root sqrt(2 u)
    assert abs(lb.self_similar_amplitude(1e-4) / (1e-4 / np.sqrt(np.pi)) - 1) < 1e-3
    assert abs(lb.self_similar_amplitude(1e5) / np.sqrt(2e5) - 1) < 1e-2


def test_latent_book_pde_matches_closed_form():
    for u in (0.1, 3.0):
        t, y = lb.simulate_metaorder(u, 1.0, n_x=4001, n_t=2000)
        assert abs(y[-1] / lb.self_similar_amplitude(u) - 1) < 0.01
        # self-similarity: y_t / sqrt(t) is constant during execution
        j = np.searchsorted(t, 0.25)
        assert abs(y[j] / np.sqrt(t[j]) / lb.self_similar_amplitude(u) - 1) < 0.02


def test_latent_book_linear_response_decay():
    t, y = lb.simulate_metaorder(0.01, 1.0, t_end=3.0, n_x=4001, n_t=2000)
    lin = lb.linear_response_path(t, 0.01, 1.0, 1.0, 1.0)
    j = np.searchsorted(t, 2.5)
    assert abs(y[j] / lin[j] - 1) < 0.02


# ---------------------------------------------------------------- Hawkes extensions
def test_power_law_weights_sum_to_n():
    w, b = hk.power_law_weights(0.7, 0.2, 0.1, 1e-3)
    assert abs(w.sum() - 0.7) < 1e-12 and np.all(np.diff(b) > 0)


def test_sum_exp_hawkes_mean_rate():
    w, b = hk.power_law_weights(0.5, 0.5, 0.1, 0.1)       # memory of about 10 s
    m = hk.SumExpHawkes1D(mu=0.5, w=w, beta=b)
    ev = m.simulate(2.2e4, rng=1)
    ev = ev[ev > 2000]
    rate = len(ev) / 2e4
    assert abs(rate / 1.0 - 1) < 0.05                      # mu / (1 - n) = 1


def test_exp_fit_recovers_branching_and_poisson_gives_zero():
    true = hk.ExpHawkes(mu=[0.5], alpha=[[1.0]], beta=[[2.0]])
    t, _ = true.simulate(8000, rng=3)
    _, n, beta, se = hk.fit_exp_hawkes_1d(t, 8000)
    assert abs(n - 0.5) < 4 * se
    tp = np.sort(np.random.default_rng(4).uniform(0, 8000, 4000))
    assert hk.fit_exp_hawkes_1d(tp, 8000)[1] < 0.05


def test_time_change_of_inhomogeneous_poisson_is_unit_rate():
    rate = lambda s: 0.5 + 0.4 * np.sin(s / 500.0)          # noqa: E731
    t = hk.simulate_inhomogeneous_poisson(rate, 20000, 0.9, rng=5)
    grid = np.linspace(0, 20000, 40001)
    tt, Tt = hk.time_change(t, rate, grid)
    gaps = np.diff(tt)
    assert abs(gaps.mean() - 1) < 4 / np.sqrt(len(gaps))


# ---------------------------------------------------------------- queue-reactive
def test_intensity_mle_recovers_rates():
    first, _ = qr.hlr_like_rates()
    w, q, k, c = qr.simulate_queue_events(first, 5e4, q0=5, rng=6)
    est = qr.estimate_intensities(w, q, k, 12, c)
    n = np.arange(13)
    for name in ("limit", "cancel", "market"):
        ok = (est["count"] > 200) & (est["se_" + name] > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (est[name] - getattr(first, name)(n)) / est["se_" + name]
        assert np.all(np.abs(z[ok]) < 4.5)


def test_mean_reversion_ratio_reference_values():
    rng = np.random.default_rng(7)
    s = rng.choice([-1, 1], 200_000)
    assert abs(qr.mean_reversion_ratio(s) - 0.5) < 0.01
    assert qr.mean_reversion_ratio(np.tile([1, -1], 100)) == 0.0


def test_model3_full_reinit_has_no_memory():
    first, second = qr.hlr_like_rates()
    p = qr.Model3Params(first, second, theta=0.7, theta_reinit=1.0)
    _, sg, _ = qr.simulate_model3(p, 3e4, rng=8)
    assert abs(qr.mean_reversion_ratio(sg) - 0.5) < 0.08


# ---------------------------------------------------------------- propagator estimation
def test_propagator_estimation_recovers_kernel():
    rng = np.random.default_rng(9)
    eps = im.long_memory_signs(2 ** 17 + 1, 0.5, rng)
    P = im.propagator_price(eps, 1.0, 0.25)
    dP = np.diff(P) + 0.5 * rng.standard_normal(2 ** 17)
    G = im.estimate_propagator(eps[:-1], dP, 1000)
    l = np.array([1, 2, 5, 10, 50])
    assert np.max(np.abs(G[l - 1] / l ** -0.25 - 1)) < 0.03


# ---------------------------------------------------------------- fills and Glosten-Milgrom
def test_fill_intensity_mle_and_naive_bias():
    d = np.repeat(np.arange(0.0, 16.0), 400)
    tau, f = mm.simulate_quote_spells(0.9, 0.3, d, 30.0, rng=10)
    A, k, cov = mm.fit_fill_intensity(d, tau, f)
    assert abs(k - 0.3) < 4 * np.sqrt(cov[1, 1])
    assert abs(np.log(A / 0.9)) < 4 * np.sqrt(cov[0, 0])
    assert mm.fit_fill_ratio_naive(d, f, 30.0)[1] < 0.2      # censoring flattens the curve


def test_glosten_milgrom_spread_and_zero_profit():
    for a in (0.1, 0.4):
        bid, ask, _, _ = gm.quotes(0.5, a)
        assert abs((ask - bid) - a) < 1e-12
    sim = gm.simulate(0.3, 30, 20000, rng=11)
    allm, u, i = gm.markouts_vs_value(sim)
    assert abs(allm) < 3e-3 and u > 0 > i


# ---------------------------------------------------------------- book estimation, micro-price, LOBSTER
def _book(n=60_000, seed=12):
    p = ml.power_law_params(L=30, lam1=1.9, a=0.5, theta0=0.5, mu=1.0)
    return p, ml.simulate_markov_lob(p, n, rng=seed, burn_in=5000, record_levels=40)


def test_rate_estimation_recovers_cst_rates():
    p, rec = _book()
    est = ml.estimate_rates(rec, 30)
    assert np.all(np.abs(est["lam"] - p.lam) < 4.5 * est["se_lam"])
    assert np.all(np.abs(est["theta"] - p.theta) < 4.5 * est["se_theta"])
    assert abs(est["mu"] - p.mu) < 4 * est["se_mu"]


def test_microprice_is_odd_in_imbalance():
    _, rec = _book(80_000, 13)
    mid = 0.5 * (rec.best_bid + rec.best_ask)
    idx, _ = mp.states(rec.q_bid, rec.q_ask, rec.best_ask - rec.best_bid, 10, 2)
    G = mp.fit(idx, mid, 10, 2)["G"]
    ok = np.isfinite(G) & np.isfinite(G[mp.mirror(np.arange(20), 10)])
    assert np.allclose(G[ok], -G[mp.mirror(np.arange(20), 10)][ok], atol=1e-10)
    assert G[9] > 0 > G[0]


def test_lobster_round_trip():
    _, rec = _book(3000, 14)
    with tempfile.TemporaryDirectory() as tmp:
        m, b = lobster.write(rec, tmp, levels=10)
        r2 = lobster.to_record(lobster.read(m, b), K=40)
    assert np.array_equal(r2.best_ask - r2.best_ask[0], rec.best_ask - rec.best_ask[0])
    assert np.array_equal(r2.q_bid, rec.q_bid) and np.array_equal(r2.event, rec.event)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
