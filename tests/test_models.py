"""Checks of each model against a result known independently.

Run with  pytest  or, without pytest,  python tests/test_models.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stochastic_lob import best_queues as bq  # noqa: E402
from stochastic_lob import hawkes, impact, market_making as mm, markov_lob, queue_reactive as qr  # noqa: E402


# ---------------------------------------------------------------- best queues
def test_p_up_symmetry_and_half():
    for q in (1, 3, 7):
        assert abs(bq.p_up_balanced(q, q) - 0.5) < 1e-8
    assert abs(bq.p_up_balanced(2, 5) + bq.p_up_balanced(5, 2) - 1) < 1e-8


def test_p_up_closed_form_matches_linear_system():
    H = bq.p_up_linear_system(6, 1.0, 1.0)
    for b in range(1, 7):
        for a in range(1, 7):
            assert abs(bq.p_up_balanced(b, a) - H[b, a]) < 2e-3


def test_p_up_quadrature_unbalanced():
    lam, nu = 0.7, 1.0
    H = bq.p_up_linear_system(5, lam, nu)
    for b, a in [(1, 1), (1, 4), (3, 2), (5, 5)]:
        assert abs(bq.p_up(b, a, lam, nu) - H[b, a]) < 2e-3


def test_p_up_monte_carlo():
    tau, up = bq.simulate_durations(3, 2, 1.0, 1.0, 20000, t_max=1e4, rng=1)
    done = np.isfinite(tau)
    p_hat = up[done].mean()
    se = np.sqrt(p_hat * (1 - p_hat) / done.sum())
    assert abs(p_hat - bq.p_up_balanced(3, 2)) < 4 * se


def test_first_passage_density_normalized():
    # Balanced walk: absorption is certain, so the density integrates to 1.
    assert abs(bq.survival(1e9, 3, 1.0, 1.0) - 0.0) < 1e-3
    # Up-drift: P(absorbed) = (nu/lam)^x.
    assert abs((1 - bq.survival(1e9, 2, 2.0, 1.0)) - 0.25) < 1e-3


def test_arctan_limit():
    assert abs(bq.p_up_balanced(40, 20) - bq.p_up_arctan(40, 20)) < 0.01


def test_duration_tail():
    t = 2e4
    ratio = bq.duration_survival(t, 2, 3, 1.0, 1.0) / bq.duration_tail_balanced(t, 2, 3, 1.0)
    assert abs(ratio - 1) < 0.02


# ---------------------------------------------------------------- queue reactive
def test_queue_reactive_stationary_law():
    rates = qr.illustrative_rates()
    pi = qr.stationary_distribution(rates, 80)
    assert pi[-1] < 1e-12
    occ, n = qr.simulate_queue(rates, 2e4, rng=3)
    # Total variation distance between occupation measure and pi.
    assert 0.5 * np.abs(occ[: len(pi)] - pi).sum() < 0.03


# ---------------------------------------------------------------- Hawkes
def test_hawkes_stationary_rate():
    m = hawkes.ExpHawkes([0.4], [[0.8]], [[2.0]])
    ts, _ = m.simulate(20000, rng=5)
    Lam = m.stationary_intensity()[0]           # 0.4 / (1 - 0.4) = 2/3
    assert abs(len(ts) / 20000 - Lam) < 0.05 * Lam


def test_hawkes_mle_recovers_parameters():
    true = hawkes.ExpHawkes([0.5], [[1.0]], [[2.5]])
    ts, ty = true.simulate(4000, rng=7)
    fit, se, _ = hawkes.fit_exp_hawkes(ts, ty, 4000, 1)
    for est, err, tru in [(fit.mu[0], se.mu[0], 0.5), (fit.alpha[0, 0], se.alpha[0, 0], 1.0),
                          (fit.beta[0, 0], se.beta[0, 0], 2.5)]:
        assert abs(est - tru) < 4 * err


def test_time_rescaling_residuals_exponential():
    m = hawkes.ExpHawkes([0.3, 0.3], [[0.6, 0.4], [0.2, 0.7]], [[2.0, 1.5], [1.5, 2.0]])
    ts, ty = m.simulate(3000, rng=2)
    for r in m.time_rescaled_residuals(ts, ty):
        assert stats.kstest(r, "expon").pvalue > 0.01


def test_signature_plot_limits_and_simulation():
    mu, a_s, a_c, b = 0.5, 0.1, 0.6, 1.0
    C = hawkes.signature_plot_theory(np.array([1e-8, 1e8]), mu, a_s, a_c, b)
    Lam = mu / (1 - (a_s + a_c) / b)
    g = (a_s - a_c) / b
    assert abs(C[0] - 2 * Lam) < 1e-6
    assert abs(C[1] - 2 * Lam / (1 - g) ** 2) < 1e-6
    m = hawkes.ExpHawkes([mu, mu], [[a_s, a_c], [a_c, a_s]], [[b, b], [b, b]])
    ts, ty = m.simulate(20000, rng=3)
    emp = hawkes.realized_signature(ts, ty, 20000, [1.0])[0]
    assert abs(emp / hawkes.signature_plot_theory(1.0, mu, a_s, a_c, b) - 1) < 0.05


# ---------------------------------------------------------------- impact
def test_fgn_covariance():
    x = impact.fgn(2 ** 16, 0.75, rng=0)
    assert abs(np.var(x) - 1) < 0.05
    assert abs(impact.autocorrelation(x, [1])[0] - impact.fgn_autocovariance(2, 0.75)[1]) < 0.03


def test_propagator_critical_exponent():
    gamma = 0.5
    eps = impact.long_memory_signs(2 ** 19, gamma, rng=1)
    lags = np.unique(np.geomspace(50, 4000, 12).astype(int))
    for zeta in (0.1, impact.critical_zeta(gamma), 0.4):
        D = impact.diffusion_signature(impact.propagator_price(eps, zeta=zeta), lags)
        slope = np.polyfit(np.log(lags), np.log(D), 1)[0]
        assert abs(slope - (1 - gamma - 2 * zeta)) < 0.1


def test_ofi_definition():
    # Bid queue grows by 3 at an unchanged price: e = +3.
    e = impact.ofi_contributions(np.array([99, 99]), np.array([5, 8]),
                                 np.array([100, 100]), np.array([4, 4]))
    assert e[0] == 3
    # Ask price drops by one tick with 2 new orders there: e = -2.
    e = impact.ofi_contributions(np.array([99, 99]), np.array([5, 5]),
                                 np.array([101, 100]), np.array([4, 2]))
    assert e[0] == -2


# ---------------------------------------------------------------- market making
def test_as_quotes_limits():
    p = mm.MMParams()
    db, da = mm.as_quotes(100.0, 0.0, p.T, p)
    # At the horizon with no inventory: half-spread (1/gamma) ln(1 + gamma/k).
    assert abs(db - np.log1p(p.gamma / p.k) / p.gamma) < 1e-12
    assert abs(db - da) < 1e-12
    # gamma -> 0: half-spread tends to 1/k.
    p0 = mm.MMParams(gamma=1e-8)
    assert abs(mm.as_quotes(100.0, 0.0, p0.T, p0)[0] - 1 / p0.k) < 1e-6


def test_inventory_strategy_reduces_risk():
    p = mm.MMParams()
    r_inv = mm.simulate(mm.as_quotes, p, 1000, rng=0)
    r_sym = mm.simulate(mm.symmetric_quotes, p, 1000, rng=0)
    assert r_inv["profit"].std() < r_sym["profit"].std()
    assert r_inv["q_T"].std() < r_sym["q_T"].std()


# ---------------------------------------------------------------- Markov book
def test_markov_lob_runs_and_is_consistent():
    p = markov_lob.power_law_params(L=20)
    rec = markov_lob.simulate_markov_lob(p, 5000, rng=0, burn_in=1000)
    assert np.all(rec.best_ask > rec.best_bid)
    assert abs(sum(rec.spread_hist.values()) - 1) < 1e-9
    assert rec.dropped_orders == 0


if __name__ == "__main__":
    failed = 0
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc!r}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
