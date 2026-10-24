"""Run the repository's estimators on one day of LOBSTER order book data.

    python examples/lobster_pipeline.py MESSAGE.csv ORDERBOOK.csv [--tick 0.01] [--lot 100] [--out lobster_report]
    python examples/lobster_pipeline.py --simulate              # no data: make a LOBSTER day from the simulator

LOBSTER (lobsterdata.com) offers free sample files, for example
AMZN_2012-06-21_34200000_57600000_message_10.csv with its orderbook file.
Download them yourself and pass the two paths. Nothing in this repository
ships or fetches market data.

What it estimates (each with a standard error), and the deck that explains it:
  1. Markovian book rates lam(i), theta(i), mu .................. deck 1
  2. P(next mid move up | q_b, q_a), clustered errors, CdL fit ... deck 2
  3. Exponential Hawkes model of market buys and sells .......... deck 4
  4. OFI and multi-level OFI regressions, out of sample ......... deck 6
  5. Micro-price adjustment by imbalance ......................... deck 6
  6. Markouts of passive fills ................................... deck 8

Caveats for real data: sizes are rounded to lots; hidden executions (type 5)
are counted as market orders although they do not touch the visible queue;
cancellation rates are biased beyond the visible depth (deck 1); the Hawkes
fit ignores intraday seasonality unless you pass --deseason (deck 4).
Output: <out>/summary.json and a few PNG figures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stochastic_lob import best_queues as bq  # noqa: E402
from stochastic_lob import hawkes as hk  # noqa: E402
from stochastic_lob import impact as im  # noqa: E402
from stochastic_lob import lobster, markouts, microprice as mp  # noqa: E402
from stochastic_lob import markov_lob as ml  # noqa: E402
from stochastic_lob.plotting import WIDE, panel_label, plt, save, use_style  # noqa: E402


def load(args):
    if args.simulate:
        p = ml.power_law_params(L=30, lam1=1.9, a=0.5, theta0=0.5, mu=1.0)
        rec = ml.simulate_markov_lob(p, 200_000, rng=7, burn_in=10_000, record_levels=40)
        folder = Path(args.out) / "simulated_lobster"
        msg, book = lobster.write(rec, folder, stem="SIM_2026-10-23", levels=10)
        print(f"simulated LOBSTER files in {folder}")
    else:
        msg, book = args.message, args.orderbook
    d = lobster.read(msg, book, max_rows=args.max_rows)
    return lobster.to_record(d, tick=args.tick, lot=args.lot, K=args.levels), d


def book_summary(rec):
    s = (rec.best_ask - rec.best_bid).astype(float)
    ok = np.abs(s) < 1e6
    dt = np.diff(rec.t)
    w = dt[ok[:-1]]
    sp = s[:-1][ok[:-1]]
    counts = {name: int(np.sum(rec.event == c)) for c, name in enumerate(ml.EVENT_NAMES)}
    return {"hours": float(rec.t[-1] / 3600), "events": int(len(rec.t)), "event_counts": counts,
            "mean_spread_ticks": float(np.sum(sp * w) / np.sum(w)),
            "share_one_tick": float(np.sum(w[sp == 1]) / np.sum(w))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("message", nargs="?")
    ap.add_argument("orderbook", nargs="?")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--tick", type=float, default=0.01, help="tick size (dollars)")
    ap.add_argument("--lot", type=int, default=100, help="shares per order unit")
    ap.add_argument("--levels", type=int, default=40, help="tick-grid depth to reconstruct")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--L", type=int, default=10, help="distances i = 1..L for rate estimation")
    ap.add_argument("--hawkes-max", type=int, default=20000, help="max market orders in the Hawkes fit")
    ap.add_argument("--deseason", action="store_true", help="time-change market orders by 30-min profile")
    ap.add_argument("--out", default="lobster_report")
    args = ap.parse_args()
    if not args.simulate and (args.message is None or args.orderbook is None):
        ap.error("give MESSAGE and ORDERBOOK files, or --simulate")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    use_style()
    rec, _ = load(args)
    report = {"book": book_summary(rec)}
    print(json.dumps(report["book"], indent=2))

    # 1. Markovian book rates --------------------------------------------------------
    est = ml.estimate_rates(rec, args.L)
    report["rates"] = {k: np.asarray(v).tolist() for k, v in est.items()
                       if k in ("lam", "se_lam", "theta", "se_theta", "mu", "se_mu")}

    # 2. P(up | queues) with clustered errors, and a constant-rate (CdL) fit ----------
    P, N, SE = ml.empirical_p_up(rec, 8)
    s1 = (rec.best_ask - rec.best_bid)[:-1] == 1
    T1 = float(np.sum(np.diff(rec.t)[s1]))
    ev, px = rec.event[1:][s1], rec.event_price[1:][s1]
    own = np.where(np.isin(ev, (0, 4)), rec.best_bid[:-1][s1], rec.best_ask[:-1][s1])
    lam_hat = float(np.sum(np.isin(ev, (0, 1)) & (px == own))) / (2 * T1)
    nu_hat = float(np.sum(np.isin(ev, (2, 3)) | (np.isin(ev, (4, 5)) & (px == own)))) / (2 * T1)
    cells = [(b, a) for b in range(1, 9) for a in range(1, 9) if N[b, a] >= 200]
    report["p_up"] = {"lam_hat": lam_hat, "nu_hat": nu_hat,
                      "cells": [{"q_b": b, "q_a": a, "p": float(P[b, a]), "se": float(SE[b, a]),
                                 "cdl": bq.p_up(b, a, lam_hat, nu_hat)} for b, a in cells]}

    # 3. Hawkes model of market buys and sells ----------------------------------------
    k = np.flatnonzero(np.isin(rec.event, (2, 3)))[: args.hawkes_max]
    times, types = rec.t[k], (rec.event[k] == 3).astype(np.int64)
    times, uniq = np.unique(times, return_index=True)       # drop same-time duplicates
    types = types[uniq]
    T = float(times[-1]) + 1e-6
    if args.deseason:
        edges = np.linspace(0, T, 14)
        rate = np.histogram(times, edges)[0] / np.diff(edges)
        prof = lambda s: np.interp(s, 0.5 * (edges[1:] + edges[:-1]), rate) / rate.mean()  # noqa: E731
        times, T = hk.time_change(times, prof, np.linspace(0, T, 20001))
    model, se, _ = hk.fit_exp_hawkes(times, types, T, 2)
    res = model.time_rescaled_residuals(times, types)
    from scipy import stats
    report["hawkes"] = {"mu": model.mu.tolist(), "alpha": model.alpha.tolist(), "beta": model.beta.tolist(),
                        "spectral_radius": model.spectral_radius,
                        "ks_p": [float(stats.kstest(r, "expon").pvalue) for r in res],
                        "n_events": int(len(times)), "deseasonalized": bool(args.deseason)}

    # 4. OFI regressions --------------------------------------------------------------
    ofi_lv, dmid = im.aggregate_multilevel_ofi(rec.t, rec.best_bid, rec.bid_depth[:, :5],
                                               rec.best_ask, rec.ask_depth[:, :5], 10.0)
    fit1 = im.ols(ofi_lv[:, 0], dmid)
    report["ofi"] = {"window_s": 10.0, "n": int(len(dmid)), "slope": fit1["slope"], "slope_se": fit1["slope_se"],
                     "r2_in_sample": fit1["r2"],
                     "r2_oos_best": float(im.out_of_sample_r2(ofi_lv[:, :1], dmid)),
                     "r2_oos_5_levels": float(im.out_of_sample_r2(ofi_lv, dmid))}

    # 5. Micro-price ------------------------------------------------------------------
    mid = 0.5 * (rec.best_bid + rec.best_ask).astype(float)
    spr = (rec.best_ask - rec.best_bid).astype(float)
    idx, imb = mp.states(rec.q_bid, rec.q_ask, spr, 10, 2)
    G = mp.fit(idx, mid, 10, 2)["G"]
    report["microprice"] = {"G_spread1": G[:10].tolist(), "G_spread2": G[10:].tolist()}

    # 6. Markouts ---------------------------------------------------------------------
    hs = np.array([0.0, 1.0, 5.0, 30.0, 120.0])
    f, M = markouts.markout_matrix(rec, hs)
    mean, err, n = markouts.mean_with_se(M)
    report["markouts"] = {"horizons_s": hs.tolist(), "mean_ticks": mean.tolist(), "se": err.tolist(),
                          "n": n.tolist()}

    # figures ------------------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    i = np.arange(1, args.L + 1)
    a1.errorbar(i, est["lam"], yerr=2 * est["se_lam"], fmt="o", ms=3, label=r"$\hat\lambda(i)$")
    a1.set_yscale("log")
    a1.set_xlabel(r"Distance to Opposite Best Quote $i$ (ticks)")
    a1.set_ylabel(r"Limit Order Rate (s$^{-1}$)")
    a1.legend(loc="upper right")
    panel_label(a1, "(a)")
    a2.errorbar(hs, mean, yerr=2 * err, fmt="o-", ms=3, label="All Passive Fills")
    a2.axhline(0, color="0.4", ls=":")
    a2.set_xscale("symlog", linthresh=1.0)
    a2.set_xlabel(r"Horizon $h$ (s)")
    a2.set_ylabel(r"Markout $M(h)$ (ticks)")
    a2.legend(loc="upper right")
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "rates_and_markouts", out)

    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
