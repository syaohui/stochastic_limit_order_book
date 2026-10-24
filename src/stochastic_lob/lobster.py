"""Read and write order book data in the LOBSTER format.

LOBSTER (lobsterdata.com) distributes NASDAQ order book data as two CSV
files per stock and day, with one row per order book event:

message file:   time, type, order_id, size, price, direction
    time        seconds after midnight
    type        1 new limit order, 2 partial cancellation, 3 full deletion,
                4 execution of a visible order, 5 execution of a hidden order,
                6 cross trade, 7 trading halt
    size        shares
    price       dollar price times 10000
    direction   +1 buy limit order, -1 sell limit order; for executions, the
                side of the resting order (so -1 means a buyer-initiated trade)

orderbook file: ask price 1, ask size 1, bid price 1, bid size 1, ask price 2, ...
    the book after the event on the same row of the message file; empty
    levels carry price 9999999999 (ask) or -9999999999 (bid) and size 0.

This module turns such files into the arrays the rest of the package uses
(best quotes in ticks, queue sizes in lots, depth on a tick grid), and
writes simulated books in the same format, so one analysis script runs on
both. The free LOBSTER sample files (for example
AMZN_2012-06-21_34200000_57600000_message_10.csv) can be dropped in directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .markov_lob import LOBRecord

EMPTY_ASK = 9999999999
EMPTY_BID = -9999999999


@dataclass
class LobsterData:
    time: np.ndarray        # seconds after midnight
    type: np.ndarray        # LOBSTER event type 1..7
    order_id: np.ndarray
    size: np.ndarray        # shares
    price: np.ndarray       # dollars
    direction: np.ndarray   # +1 buy order, -1 sell order
    ask_p: np.ndarray       # (n, K) dollars, NaN when the level is empty
    ask_q: np.ndarray       # (n, K) shares
    bid_p: np.ndarray       # (n, K) dollars, NaN when the level is empty
    bid_q: np.ndarray       # (n, K) shares


def read(message_path, orderbook_path, max_rows=None) -> LobsterData:
    """Load one LOBSTER message file and its orderbook file."""
    msg = np.loadtxt(message_path, delimiter=",", max_rows=max_rows, ndmin=2)
    book = np.loadtxt(orderbook_path, delimiter=",", max_rows=max_rows, ndmin=2)
    if len(msg) != len(book):
        raise ValueError("message and orderbook files must have the same number of rows")
    if book.shape[1] % 4:
        raise ValueError("orderbook file must have 4 columns per level")
    ap, aq, bp, bq = (book[:, j::4] for j in range(4))
    ap = np.where((aq > 0) & (ap < EMPTY_ASK), ap / 1e4, np.nan)
    bp = np.where((bq > 0) & (bp > EMPTY_BID), bp / 1e4, np.nan)
    return LobsterData(time=msg[:, 0], type=msg[:, 1].astype(int), order_id=msg[:, 2].astype(np.int64),
                       size=msg[:, 3], price=msg[:, 4] / 1e4, direction=np.sign(msg[:, 5]).astype(int),
                       ask_p=ap, ask_q=aq, bid_p=bp, bid_q=bq)


def event_codes(d: LobsterData) -> np.ndarray:
    """Map LOBSTER events to markov_lob.EVENT_NAMES codes (-1 for other events).

    0 limit buy, 1 limit sell, 2 market buy (a sell order executed),
    3 market sell, 4 cancel buy, 5 cancel sell.
    """
    buy = d.direction > 0
    code = np.full(len(d.type), -1, dtype=np.int8)
    code[(d.type == 1) & buy] = 0
    code[(d.type == 1) & ~buy] = 1
    execd = np.isin(d.type, (4, 5))
    code[execd & ~buy] = 2
    code[execd & buy] = 3
    canc = np.isin(d.type, (2, 3))
    code[canc & buy] = 4
    code[canc & ~buy] = 5
    return code


def to_record(d: LobsterData, tick=0.01, lot=100, K=None) -> LOBRecord:
    """Convert to a LOBRecord: prices in ticks, sizes in lots, depth on the tick grid.

    bid_depth[k, m] is the size (lots) at p_B - m ticks and ask_depth[k, m] at
    p_A + m ticks, with empty ticks set to zero. K defaults to the number of
    levels in the file. Rows with an empty side get sentinel quotes, as in
    simulate_markov_lob. depth_profile and spread_hist are left empty.
    """
    n, levels = d.ask_p.shape
    K = levels if K is None else K
    ap_t = np.rint(d.ask_p / tick)
    bp_t = np.rint(d.bid_p / tick)
    best_a = np.where(np.isfinite(ap_t[:, 0]), ap_t[:, 0], np.iinfo(np.int64).max)
    best_b = np.where(np.isfinite(bp_t[:, 0]), bp_t[:, 0], np.iinfo(np.int64).min)
    bid_depth = np.zeros((n, K), dtype=np.int64)
    ask_depth = np.zeros((n, K), dtype=np.int64)
    rows = np.arange(n)
    for j in range(levels):
        ma = ap_t[:, j] - ap_t[:, 0]
        ok = np.isfinite(ma) & (ma >= 0) & (ma < K)
        ask_depth[rows[ok], ma[ok].astype(int)] += np.rint(d.ask_q[ok, j] / lot).astype(np.int64)
        mb = bp_t[:, 0] - bp_t[:, j]
        ok = np.isfinite(mb) & (mb >= 0) & (mb < K)
        bid_depth[rows[ok], mb[ok].astype(int)] += np.rint(d.bid_q[ok, j] / lot).astype(np.int64)
    return LOBRecord(
        t=d.time - d.time[0],
        best_bid=best_b.astype(np.int64),
        best_ask=best_a.astype(np.int64),
        q_bid=bid_depth[:, 0],
        q_ask=ask_depth[:, 0],
        event=event_codes(d),
        depth_profile=np.empty(0),
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        event_price=np.rint(d.price / tick).astype(np.int64),
    )


def write(rec: LOBRecord, folder, stem="SIM", tick=0.01, lot=100, price0=100.0, levels=10, t0=34200.0):
    """Write a simulated record (with bid_depth, ask_depth, event_price) as LOBSTER files.

    Levels are the first `levels` nonempty ticks on each side. Every simulated
    order has `lot` shares and order id 0 (the simulator does not track ids).
    Returns the two file paths.
    """
    if rec.bid_depth is None or rec.event_price is None:
        raise ValueError("simulate with record_levels > 0")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    n = len(rec.t)
    base = int(round(price0 / tick))
    code_type = {0: 1, 1: 1, 2: 4, 3: 4, 4: 3, 5: 3}
    code_dir = {0: 1, 1: -1, 2: -1, 3: 1, 4: 1, 5: -1}
    msg = np.column_stack([
        t0 + rec.t,
        [code_type[int(e)] for e in rec.event],
        np.zeros(n),
        np.full(n, lot),
        (base + rec.event_price) * tick * 1e4,
        [code_dir[int(e)] for e in rec.event],
    ])
    book = np.zeros((n, 4 * levels))
    book[:, 0::4] = EMPTY_ASK
    book[:, 2::4] = EMPTY_BID
    for k in range(n):
        for side, depth, best, sgn, col in ((0, rec.ask_depth[k], rec.best_ask[k], 1, 0),
                                            (1, rec.bid_depth[k], rec.best_bid[k], -1, 2)):
            ticks = np.flatnonzero(depth > 0)[:levels]
            px = (base + best + sgn * ticks) * tick * 1e4
            book[k, col + 4 * np.arange(len(ticks))] = px
            book[k, col + 1 + 4 * np.arange(len(ticks))] = depth[ticks] * lot
    mp = folder / f"{stem}_message_{levels}.csv"
    bp = folder / f"{stem}_orderbook_{levels}.csv"
    np.savetxt(mp, msg, delimiter=",", fmt=["%.9f", "%d", "%d", "%d", "%d", "%d"])
    np.savetxt(bp, book, delimiter=",", fmt="%d")
    return mp, bp
