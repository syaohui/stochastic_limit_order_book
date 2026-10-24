"""Checks of the liquidity-provider modules against results known independently.

Run with  pytest  or, without pytest,  python tests/test_liquidity.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stochastic_lob import amm, best_queues, impact, market_making as mm, markouts, markov_lob  # noqa: E402
from stochastic_lob import queue_position as qp  # noqa: E402


# ---------------------------------------------------------------- queue position
def test_state_dependent_p_up_reduces_to_cdl():
    # theta = 0 and lam = mu: constant balanced rates, the Cont-de Larrard case.
    h = qp.p_up_state_dependent(qp.QueueRates(lam=1.0, mu=1.0, theta=0.0), 40, 40)
    for c, b in [(1, 1), (2, 5), (5, 2), (4, 4)]:
        assert abs(h[c, b] - best_queues.p_up_balanced(c, b)) < 5e-3


def test_queue_value_matches_monte_carlo():
    r = qp.QueueRates(lam=1.0, mu=0.9, theta=0.01, nu=0.03, jump=2.0)
    V = qp.solve_queue_value(r, a_max=6, b_max=30, c_max=30)
    a, b, c = 5, 2, 6
    filled, bf, cf = qp.simulate_order(r, a, b, c, 30000, rng=4)
    p = filled.mean()
    assert abs(p - V.p_fill[a, b, c]) < 4 * np.sqrt(p * (1 - p) / filled.size)
    sweep = bf == -2
    cost = np.where(sweep, r.jump, 0.0)
    small = filled & ~sweep
    cost[small] = np.where(bf[small] == 0, 1.0,
                           2 * V.p_up[np.clip(cf[small], 0, 30), np.clip(bf[small], 0, 30)] - 1)
    val = np.where(filled, 0.5 - cost, 0.0)
    assert abs(val.mean() - V.value[a, b, c]) < 4 * val.std() / np.sqrt(val.size)


def test_queue_value_truncation_stable():
    r = qp.QueueRates(lam=1.0, mu=0.9, theta=0.01, nu=0.03, jump=2.0)
    V1 = qp.solve_queue_value(r, a_max=4, b_max=25, c_max=25)
    V2 = qp.solve_queue_value(r, a_max=4, b_max=35, c_max=35)
    assert abs(V1.value[4, 3, 6] - V2.value[4, 3, 6]) < 1e-3


# ---------------------------------------------------------------- GLFT exact
def _glft_params():
    return mm.MMParams(S0=0.0, T=600.0, sigma=0.3, dt=1.0, gamma=0.01, k=0.3, A=0.9)


def test_glft_terminal_quotes():
    p = _glft_params()
    db, da = mm.glft_exact_quotes(10, p, [0.0])
    base = np.log1p(p.gamma / p.k) / p.gamma
    assert np.allclose(db[0, :-1], base) and np.allclose(da[0, 1:], base)


def test_glft_stationary_matches_closed_form_near_zero_inventory():
    p = _glft_params()
    sb, sa = mm.glft_stationary_quotes(30, p)
    gb, ga = mm.glft_quotes(np.array([-1, 0, 1]), p)
    assert np.max(np.abs(sb[29:32] - gb)) < 0.01
    assert np.max(np.abs(sa[29:32] - ga)) < 0.01


def test_glft_ode():
    p = _glft_params()
    Q = 8
    M = mm.glft_matrix(Q, p)
    L0 = np.linalg.eigvalsh(M)[0]
    h = 1e-4
    v1 = mm.glft_value_vectors(Q, p, [3.0])[:, 0]
    v2 = mm.glft_value_vectors(Q, p, [3.0 + h])[:, 0]
    # The solver returns exp(L0 tau) v, so d/dtau of it is (L0 I - M) of it.
    rhs = (L0 * np.eye(2 * Q + 1) - M) @ (0.5 * (v1 + v2))
    assert np.max(np.abs((v2 - v1) / h - rhs)) < 1e-6 * np.max(np.abs(rhs))


def test_glft_drift_skews_quotes():
    p = _glft_params()
    db0, da0 = mm.glft_exact_quotes(5, p, [600.0], drift=0.0)
    db1, da1 = mm.glft_exact_quotes(5, p, [600.0], drift=0.002)
    assert db1[0, 5] < db0[0, 5] and da1[0, 5] > da0[0, 5]


# ---------------------------------------------------------------- AMM
def test_lvr_cpmm_expectation():
    sig, T = 0.05, 1.0
    lvr = amm.simulate_lvr_cpmm(sig, T, 500, 20000, rng=1)
    exact = amm.expected_lvr_cpmm(T, sig)
    assert abs(lvr.mean() - exact) < 4 * lvr.std() / np.sqrt(lvr.size)


def test_lvr_concentrated_formula():
    sig, w, T = 0.02, 1.5, 1.0
    lvr, inside = amm.simulate_lvr_range(sig, T, 500, 20000, 1 / w, w, rng=2)
    V0 = amm.range_position(1.0, 1 / w, w)[2]
    # Over a short horizon the price stays near 1, so LVR ~ rate * V0 * T.
    pred = amm.lvr_per_capital_concentrated(sig, w) * V0 * T
    assert inside.mean() > 0.999
    assert abs(lvr.mean() / pred - 1) < 0.02


def test_fee_band_reduces_arbitrage():
    r0 = amm.simulate_cpmm_arbitrage(0.05, 1.0, 1 / 1440, 0.0, 2000, rng=3)
    r1 = amm.simulate_cpmm_arbitrage(0.05, 1.0, 1 / 1440, 0.003, 2000, rng=3)
    assert r1["arb"].mean() < 0.5 * r0["arb"].mean()


# ---------------------------------------------------------------- markouts and OFI
def _record():
    return markov_lob.simulate_markov_lob(markov_lob.power_law_params(L=20), 20000, rng=5,
                                          burn_in=2000, record_levels=4)


def test_markout_zero_horizon_identity():
    rec = _record()
    f, M = markouts.markout_matrix(rec, [0.0])
    nondep = f["q_before"] >= 2
    # A trade that does not empty the queue leaves the quotes unchanged.
    assert np.allclose(M[nondep, 0], 0.5 * f["spread_before"][nondep])


def test_multilevel_ofi_first_column_is_cks():
    rec = _record()
    e = impact.multilevel_ofi_contributions(rec.best_bid, rec.bid_depth, rec.best_ask, rec.ask_depth)
    e0 = impact.ofi_contributions(rec.best_bid, rec.q_bid, rec.best_ask, rec.q_ask)
    assert np.allclose(e[:, 0], e0)


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
