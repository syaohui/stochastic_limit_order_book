"""Markovian limit order book (Cont, Stoikov and Talreja 2010).

The book lives on a price grid with tick size delta (price units). The state is
the vector X, where X[p] is the signed number of resting unit orders at grid
level p: X[p] > 0 counts sell orders (asks) and X[p] < 0 counts buy orders
(bids). The best ask p_A is the lowest level with X > 0 and the best bid p_B is
the highest level with X < 0.

With i >= 1 the distance in ticks to the OPPOSITE best quote, the events are

    limit buy  at p_A - i     rate lam[i-1]                (orders / s)
    limit sell at p_B + i     rate lam[i-1]
    market buy                rate mu, removes one ask at p_A
    market sell               rate mu, removes one bid at p_B
    cancel one buy  at p_A-i  rate theta[i-1] * |X[p]|     (theta in 1/s)
    cancel one sell at p_B+i  rate theta[i-1] * X[p]

for 1 <= i <= L, with L = len(lam). Resting orders further than L ticks from
the opposite best quote are cancelled at rate theta[L-1] per order.

The zero-intelligence model of Smith, Farmer, Gillemot and Krishnamurthy (2003)
in its windowed form is the special case lam[i] = const, theta[i] = const.

The simulation is exact (Gillespie 1977): waiting times are exponential with
the total event rate, and the event is drawn in proportion to its rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MarkovLOBParams:
    """Rates of the Markovian book. Time unit is the second."""

    lam: np.ndarray          # limit order rate at distance i = 1..L  (orders / s)
    theta: np.ndarray        # cancellation rate per order at distance i (1 / s)
    mu: float                # market order rate per side              (orders / s)
    tick: float = 0.01       # tick size delta                          (price)

    def __post_init__(self):
        self.lam = np.asarray(self.lam, dtype=float)
        self.theta = np.asarray(self.theta, dtype=float)
        if self.lam.shape != self.theta.shape:
            raise ValueError("lam and theta must have the same length L")
        if np.any(self.lam < 0) or np.any(self.theta <= 0) or self.mu < 0:
            raise ValueError("rates must be nonnegative and theta positive")

    @property
    def L(self) -> int:
        return len(self.lam)

    def far_depth(self) -> np.ndarray:
        """Depth lam(i)/theta(i) of an M/M/infinity queue at fixed distance i.

        This is the mean depth a level would have if no market order reached
        it and the quotes did not move. It is a reference curve, not an exact
        result for the full book.
        """
        return self.lam / self.theta


def power_law_params(L=30, lam1=1.9, a=0.5, theta0=0.5, mu=1.0, tick=0.01):
    """CST-type rates: lam(i) = lam1 * i^(-a), constant theta (paper Eq. 2).

    The default numbers are illustrative. They give a book a few orders deep
    near the quotes, like the order of magnitude CST report for liquid stocks,
    but they are not fitted to any data set.
    """
    i = np.arange(1, L + 1)
    return MarkovLOBParams(lam=lam1 * i ** (-a), theta=np.full(L, theta0), mu=mu, tick=tick)


def zero_intelligence_params(L=30, lam0=1.0, theta0=0.2, mu=2.0, tick=0.01):
    """Windowed zero-intelligence model: uniform limit-order placement over L levels."""
    return MarkovLOBParams(lam=np.full(L, lam0), theta=np.full(L, theta0), mu=mu, tick=tick)


@dataclass
class LOBRecord:
    """Output of :func:`simulate_markov_lob`.

    Arrays indexed by event k hold the state just AFTER event k. Prices are in
    ticks relative to the initial mid price.
    """

    t: np.ndarray            # event times (s)
    best_bid: np.ndarray     # p_B (ticks)
    best_ask: np.ndarray     # p_A (ticks)
    q_bid: np.ndarray        # orders at the best bid
    q_ask: np.ndarray        # orders at the best ask
    event: np.ndarray        # event code, see EVENT_NAMES
    depth_profile: np.ndarray  # time-averaged depth at distance i=1..L from opposite quote, (L,)
    spread_hist: dict = field(default_factory=dict)  # spread (ticks) -> time fraction
    dropped_orders: int = 0  # orders discarded by grid recentring (should be ~0)
    bid_depth: np.ndarray | None = None  # (n, K) orders at p_B - m ticks, m = 0..K-1
    ask_depth: np.ndarray | None = None  # (n, K) orders at p_A + m ticks, m = 0..K-1
    event_price: np.ndarray | None = None  # price level (ticks) touched by event k


EVENT_NAMES = ("limit_buy", "limit_sell", "market_buy", "market_sell", "cancel_buy", "cancel_sell")


def _best_quotes(x):
    asks = np.flatnonzero(x > 0)
    bids = np.flatnonzero(x < 0)
    pa = asks[0] if asks.size else -1
    pb = bids[-1] if bids.size else -1
    return pb, pa


def _pick(weights, total, rng):
    """Index drawn with probability weights[i] / total."""
    i = int(np.searchsorted(np.cumsum(weights), rng.random() * total, side="right"))
    return min(i, len(weights) - 1)


def simulate_markov_lob(params: MarkovLOBParams, n_events: int, rng=None,
                        init_depth: int | None = None, burn_in: int = 0,
                        record_levels: int = 0) -> LOBRecord:
    """Simulate the Markovian book for ``n_events`` events after ``burn_in`` events.

    The book starts with spread one tick and ``init_depth`` orders at each of the
    L levels on each side (default: rounded lam/theta, at least 1).

    With record_levels = K > 0, the depth at the K price levels from each best
    quote outward (empty levels included) is stored after every event. This is
    what multi-level order flow imbalance needs.
    """
    rng = np.random.default_rng(rng)
    L = params.L
    lam, theta, mu = params.lam, params.theta, params.mu
    N = 8 * L + 16                       # grid size in ticks
    x = np.zeros(N, dtype=np.int64)
    centre = N // 2
    if init_depth is None:
        init = np.maximum(1, np.rint(params.far_depth())).astype(np.int64)
    else:
        init = np.full(L, int(init_depth), dtype=np.int64)
    x[centre - L:centre] = -init[::-1]   # bids at centre-1, centre-2, ...
    x[centre:centre + L] = init          # asks at centre, centre+1, ...
    offset = -centre                     # absolute tick = index + offset; mid starts at -0.5

    idx = np.arange(N)
    total = n_events + burn_in
    t = 0.0
    out_t = np.empty(n_events)
    out_b = np.empty(n_events, dtype=np.int64)
    out_a = np.empty(n_events, dtype=np.int64)
    out_qb = np.empty(n_events, dtype=np.int64)
    out_qa = np.empty(n_events, dtype=np.int64)
    out_ev = np.empty(n_events, dtype=np.int8)
    out_px = np.empty(n_events, dtype=np.int64)
    K = int(record_levels)
    out_bd = np.zeros((n_events, K), dtype=np.int32) if K else None
    out_ad = np.zeros((n_events, K), dtype=np.int32) if K else None
    steps = np.arange(K)
    depth_acc = np.zeros(L)
    spread_acc: dict[int, float] = {}
    time_acc = 0.0
    dropped = 0

    for k in range(total):
        pb, pa = _best_quotes(x)
        if pb < 0 and pa < 0:
            raise RuntimeError("book emptied on both sides; raise lam or lower mu")
        pa_ref = pa if pa >= 0 else pb + 1   # reference quote when one side is empty
        pb_ref = pb if pb >= 0 else pa - 1

        # Distances to the opposite best quote, capped at L for cancellation.
        d_bid = np.clip(pa_ref - idx, 1, L)       # meaningful where x < 0
        d_ask = np.clip(idx - pb_ref, 1, L)       # meaningful where x > 0
        cb = np.where(x < 0, -x * theta[d_bid - 1], 0.0)
        ca = np.where(x > 0, x * theta[d_ask - 1], 0.0)
        # Orders beyond the window use theta[L-1]: the clip maps them there.

        # Limit order placements must land on the grid.
        lb_levels = pa_ref - np.arange(1, L + 1)
        ls_levels = pb_ref + np.arange(1, L + 1)
        lb_ok = lb_levels >= 0
        ls_ok = ls_levels < N
        r_lb = lam[lb_ok].sum()
        r_ls = lam[ls_ok].sum()
        r_mb = mu if pa >= 0 else 0.0
        r_ms = mu if pb >= 0 else 0.0
        r_cb = cb.sum()
        r_ca = ca.sum()
        rates = np.array([r_lb, r_ls, r_mb, r_ms, r_cb, r_ca])
        R = rates.sum()
        dt = rng.exponential(1.0 / R)

        if k >= burn_in:
            # Time-weighted statistics of the state that held during dt.
            if pa >= 0 and pb >= 0:
                s = int(pa - pb)
                spread_acc[s] = spread_acc.get(s, 0.0) + dt
                # Bid depth at p_A - i and ask depth at p_B + i, averaged over sides.
                for side_levels, sign in ((pa - np.arange(1, L + 1), -1), (pb + np.arange(1, L + 1), 1)):
                    ok = (side_levels >= 0) & (side_levels < N)
                    q = np.zeros(L)
                    q[ok] = np.maximum(sign * x[side_levels[ok]], 0)
                    depth_acc += 0.5 * q * dt
                time_acc += dt
        t += dt

        ev = _pick(rates, R, rng)
        if ev == 0:
            lvl = lb_levels[lb_ok][_pick(lam[lb_ok], r_lb, rng)]
            x[lvl] -= 1
        elif ev == 1:
            lvl = ls_levels[ls_ok][_pick(lam[ls_ok], r_ls, rng)]
            x[lvl] += 1
        elif ev == 2:
            lvl = pa
            x[pa] -= 1
        elif ev == 3:
            lvl = pb
            x[pb] += 1
        elif ev == 4:
            lvl = _pick(cb, r_cb, rng)
            x[lvl] += 1
        else:
            lvl = _pick(ca, r_ca, rng)
            x[lvl] -= 1
        lvl_abs = int(lvl) + offset          # absolute tick, before any recentring

        # Recentre the grid when the mid drifts out of the middle half.
        pb2, pa2 = _best_quotes(x)
        ref = pa2 if pa2 >= 0 else pb2
        shift = int(ref - centre)
        if abs(shift) > N // 4:
            if shift > 0:
                dropped += int(np.abs(x[:shift]).sum())
                x = np.concatenate([x[shift:], np.zeros(shift, dtype=np.int64)])
            else:
                dropped += int(np.abs(x[shift:]).sum())
                x = np.concatenate([np.zeros(-shift, dtype=np.int64), x[:shift]])
            offset += shift
            pb2 = pb2 - shift if pb2 >= 0 else -1
            pa2 = pa2 - shift if pa2 >= 0 else -1

        if k >= burn_in:
            j = k - burn_in
            out_t[j] = t
            out_b[j] = pb2 + offset if pb2 >= 0 else np.iinfo(np.int64).min
            out_a[j] = pa2 + offset if pa2 >= 0 else np.iinfo(np.int64).max
            out_qb[j] = -x[pb2] if pb2 >= 0 else 0
            out_qa[j] = x[pa2] if pa2 >= 0 else 0
            out_ev[j] = ev
            out_px[j] = lvl_abs
            if K:
                if pb2 >= 0:
                    lv = pb2 - steps
                    ok = lv >= 0
                    out_bd[j, ok] = np.maximum(-x[lv[ok]], 0)
                if pa2 >= 0:
                    lv = pa2 + steps
                    ok = lv < N
                    out_ad[j, ok] = np.maximum(x[lv[ok]], 0)

    t0 = out_t[0] if n_events else 0.0
    return LOBRecord(
        t=out_t - t0,
        best_bid=out_b,
        best_ask=out_a,
        q_bid=out_qb,
        q_ask=out_qa,
        event=out_ev,
        depth_profile=depth_acc / max(time_acc, 1e-300),
        spread_hist={s: v / time_acc for s, v in sorted(spread_acc.items())},
        dropped_orders=dropped,
        bid_depth=out_bd,
        ask_depth=out_ad,
        event_price=out_px,
    )


def empirical_p_up(rec: LOBRecord, max_q: int = 10):
    """Empirical probability that the next mid-price move is up.

    Conditions on the best-queue sizes (q_bid, q_ask) observed right after each
    event at which the spread is one tick. Returns (p, n, se): p[b, a] is the
    estimate for q_bid = b, q_ask = a, n[b, a] the number of samples and
    se[b, a] a cluster-robust standard error. Only sizes 1..max_q are kept.

    Samples are not independent: every event elsewhere in the book adds a
    sample with the same outcome until the mid moves. The standard error
    therefore treats all samples that precede the same mid move as one
    cluster: se^2 = sum_c (sum_{i in c} (y_i - p))^2 / n^2.
    """
    mid2 = rec.best_bid + rec.best_ask          # twice the mid price, in ticks
    valid = (rec.q_bid > 0) & (rec.q_ask > 0)
    moves = np.flatnonzero(np.diff(mid2) != 0) + 1
    # For each event k, the index of the next mid change strictly after k.
    nxt = np.searchsorted(moves, np.arange(len(mid2)), side="right")
    has_next = nxt < len(moves)
    up = np.zeros(len(mid2), dtype=bool)
    up[has_next] = mid2[moves[nxt[has_next]]] > mid2[np.arange(len(mid2))[has_next]]
    spread1 = (rec.best_ask - rec.best_bid) == 1
    keep = valid & has_next & spread1 & (rec.q_bid <= max_q) & (rec.q_ask <= max_q)
    n = np.zeros((max_q + 1, max_q + 1))
    s = np.zeros((max_q + 1, max_q + 1))
    np.add.at(n, (rec.q_bid[keep], rec.q_ask[keep]), 1)
    np.add.at(s, (rec.q_bid[keep], rec.q_ask[keep]), up[keep])
    with np.errstate(invalid="ignore"):
        p = s / n
    # cluster-robust variance: clusters are (cell, next mid move)
    b, a, c = rec.q_bid[keep], rec.q_ask[keep], nxt[keep]
    resid = up[keep] - p[b, a]
    key = (b * (max_q + 1) + a) * (len(moves) + 1) + c
    uk, inv = np.unique(key, return_inverse=True)
    csum = np.bincount(inv, weights=resid)
    cell = uk // (len(moves) + 1)
    var = np.bincount(cell, weights=csum ** 2, minlength=(max_q + 1) ** 2).reshape(max_q + 1, max_q + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(var) / n
    return p, n, se


def estimate_rates(rec: LOBRecord, L: int):
    """Maximum-likelihood rates of the Markovian book from an event record.

    Needs rec.event_price and multi-level depth (record_levels >= L + a few).
    For the state that holds between events k and k+1 (duration dt_k), with
    best quotes p_B, p_A and spread s, an order resting m ticks behind its own
    best quote is i = s + m ticks from the opposite one. Then, for i = 1..L:

        lam(i)   = N_limit(i) / (2 T)          per side (orders / s)
        theta(i) = N_cancel(i) / E(i)          per order (1 / s)
        mu       = N_market / (2 T)            per side (orders / s)

    with T the observed time (s) and E(i) = sum_k dt_k * (orders at distance i
    on both sides) the exposure (order-seconds). These are the Poisson MLEs;
    standard errors are sqrt(N)/denominator. Distances are measured against the
    quotes just before each event, as in Cont, Stoikov and Talreja (2010).
    """
    if rec.event_price is None or rec.bid_depth is None:
        raise ValueError("simulate with record_levels > 0; event_price is required")
    pb, pa = rec.best_bid, rec.best_ask
    ev, px = rec.event, rec.event_price
    T = rec.t[-1] - rec.t[0]
    # events 1..n-1, measured against the state after the previous event
    pb0, pa0, e, p = pb[:-1], pa[:-1], ev[1:], px[1:]
    dist = np.where(np.isin(e, (0, 4)), pa0 - p, p - pb0)
    valid = (np.abs(pa0 - pb0) < 10 ** 6)
    out = {"T": T}
    for name, codes in (("lam", (0, 1)), ("cancel", (4, 5))):
        sel = np.isin(e, codes) & valid & (dist >= 1) & (dist <= L)
        out["N_" + name] = np.bincount(dist[sel] - 1, minlength=L)[:L].astype(float)
    out["lam"] = out["N_lam"] / (2 * T)
    out["se_lam"] = np.sqrt(out["N_lam"]) / (2 * T)
    N_m = float(np.sum(np.isin(e, (2, 3))))
    out["mu"], out["se_mu"] = N_m / (2 * T), np.sqrt(N_m) / (2 * T)
    # exposure: state after event k holds for t[k+1] - t[k]
    dt = np.diff(rec.t)
    s = (pa - pb)[:-1]
    ok = np.abs(s) < 10 ** 6
    K = rec.bid_depth.shape[1]
    E = np.zeros(L + K + 2)
    m = np.arange(K)
    for depth in (rec.bid_depth[:-1], rec.ask_depth[:-1]):
        i = s[ok, None] + m[None, :]
        w = depth[ok] * dt[ok, None]
        keep = i <= L
        np.add.at(E, i[keep], w[keep])
    E = E[1:L + 1]
    out["exposure"] = E
    with np.errstate(invalid="ignore", divide="ignore"):
        out["theta"] = out["N_cancel"] / E
        out["se_theta"] = np.sqrt(out["N_cancel"]) / E
    return out
