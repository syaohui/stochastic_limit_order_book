"""E15. Estimating the Markovian book's rates from its event stream.

Simulates the power-law (CST-type) book, writes and re-reads it as LOBSTER
files (the format of public NASDAQ data), and recovers lam(i), theta(i) and
mu by Poisson maximum likelihood with standard errors. Then fits the
power-law exponent a of lam(i) = lam1 i^(-a) by weighted least squares on
log lam.

Output: figures/fig15_rate_estimation.png, results/e15_rate_estimation.{json,tex}
"""
import tempfile

import numpy as np

from common import FIG, fmt, fmt_pm, write_numbers
from stochastic_lob import lobster, markov_lob as ml
from stochastic_lob.plotting import WIDE, panel_label, plt, save, use_style

N_EVENTS = 300_000
L = 30


def main():
    use_style()
    p = ml.power_law_params(L=L, lam1=1.9, a=0.5, theta0=0.5, mu=1.0)
    rec = ml.simulate_markov_lob(p, N_EVENTS, rng=15, burn_in=20_000, record_levels=40)

    # Round trip through the LOBSTER format, as real data would arrive.
    with tempfile.TemporaryDirectory() as tmp:
        mp, bp = lobster.write(rec, tmp, levels=10)
        d = lobster.read(mp, bp)
    rec2 = lobster.to_record(d, K=40)
    rec2.t = rec.t                           # same clock, avoid float round-off in the CSV
    est_full = ml.estimate_rates(rec, L)       # full depth (simulator)
    est = ml.estimate_rates(rec2, L)           # 10 visible levels per side (LOBSTER file)

    i = np.arange(1, L + 1)
    z_lam = (est_full["lam"] - p.lam) / est_full["se_lam"]
    z_th = (est_full["theta"] - p.theta) / est_full["se_theta"]

    # Weighted least squares for log lam(i) = log lam1 - a log i, weights 1/var(log lam_hat).
    y = np.log(est_full["lam"])
    w = (est_full["lam"] / est_full["se_lam"]) ** 2
    X = np.column_stack([np.ones(L), -np.log(i)])
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    b = cov @ X.T @ (w * y)
    a_hat, a_se = b[1], np.sqrt(cov[1, 1])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    a1.loglog(i, p.lam, "-", color="0.3", label=r"True $\lambda(i) = 1.9\, i^{-0.5}$")
    a1.errorbar(i, est_full["lam"], yerr=2 * est_full["se_lam"], fmt="o", ms=3, mfc="white",
                label="MLE, Full Book")
    a1.loglog(i, np.exp(b[0]) * i ** (-a_hat), "--", label=rf"Fit, $\hat a = {a_hat:.3f}$")
    a1.set_xlabel(r"Distance to Opposite Best Quote $i$ (ticks)")
    a1.set_ylabel(r"Limit Order Rate $\lambda(i)$ (s$^{-1}$)")
    a1.set_ylim(0.2, 6)
    a1.legend(loc="upper right", fontsize=7.5)
    panel_label(a1, "(a)")

    a2.set_yscale("log")
    a2.axhline(p.theta[0], color="0.3", label=r"True $\theta$")
    a2.errorbar(i, est_full["theta"], yerr=2 * est_full["se_theta"], fmt="o", ms=3, mfc="white",
                label="Full Book")
    vis = np.isfinite(est["theta"]) & (est["exposure"] > 0)
    a2.plot(i[vis], est["theta"][vis], "s", ms=3, label="10 Levels (LOBSTER File)")
    a2.set_xlabel(r"Distance to Opposite Best Quote $i$ (ticks)")
    a2.set_ylabel(r"Cancellation Rate $\theta(i)$ (s$^{-1}$)")
    a2.set_ylim(0.2, 2e3)
    a2.legend(loc="upper left", fontsize=7.5)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig15_rate_estimation", FIG)

    # Where the 10-level file stops seeing the book: mean distance of level 10.
    lvl10 = np.nanmean(np.abs(np.rint(d.ask_p[:, 9] / 0.01) - np.rint(d.bid_p[:, 0] / 0.01)))
    # First distance at which the 10-level estimate is off by more than 10%.
    off = np.flatnonzero(~(np.abs(est["theta"] / p.theta - 1) < 0.1))
    i_bad = int(i[off[0]]) if off.size else L + 1
    write_numbers("e15_rate_estimation", {
        "EfifteenNevents": f"{N_EVENTS:,}".replace(",", "{,}"),
        "EfifteenHours": fmt(rec.t[-1] / 3600, 1),
        "EfifteenMaxZlam": fmt(np.max(np.abs(z_lam)), 1),
        "EfifteenMaxZtheta": fmt(np.max(np.abs(z_th)), 1),
        "EfifteenAhat": fmt_pm(a_hat, a_se, 3),
        "EfifteenMuhat": fmt_pm(est_full["mu"], est_full["se_mu"], 3),
        "EfifteenLevelTen": fmt(lvl10, 1),
        "EfifteenIbad": str(i_bad),
    })


if __name__ == "__main__":
    main()
