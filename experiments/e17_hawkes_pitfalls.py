"""E17. Three ways an exponential Hawkes fit misreads the branching ratio.

(a) Kernel misspecification. Events come from a Hawkes process with a slowly
    decaying (power-law-like) kernel and branching ratio n = 0.9. An
    exponential-kernel model fitted on windows of length W picks a short
    decay time 1/beta_hat and reports less than half of the true branching
    ratio, for every window length tried: one exponential cannot represent
    excitation spread over six decades of time.
(b) Seasonality. A pure Poisson process (n = 0) with a U-shaped intraday rate,
    fitted day by day, gives n_hat > 0; mapping time through the estimated
    seasonal profile removes the bias.
(c) Regime switches. A Poisson process whose rate jumps between two levels
    at random times (n = 0) gives an even larger n_hat (Filimonov and
    Sornette 2015).

Output: figures/fig17_hawkes_pitfalls.png, results/e17_hawkes_pitfalls.{json,tex}
"""
import numpy as np

from common import FIG, fmt, fmt_pm, write_numbers
from stochastic_lob import hawkes as hk
from stochastic_lob.plotting import panel_label, plt, save, use_style

N_TRUE = 0.9
EPS = 0.15            # power-law exponent of the kernel tail, phi ~ t^-(1+eps)
TAU0 = 0.1            # short-time cut-off (s)
BETA_MIN = 1e-4       # longest memory 1/BETA_MIN (s)
RATE = 0.5            # stationary event rate (1/s)
WINDOWS = (60, 300, 1800, 7200, 28800)
N_WIN = 8             # windows per length
DAY = 23400.0         # trading day (s)
N_DAYS = 12


def fit_windows(times, W, starts):
    out = []
    for s in starts:
        ev = times[(times >= s) & (times < s + W)] - s
        _, n, b, _ = hk.fit_exp_hawkes_1d(ev, W)
        out.append((n, b))
    return np.array(out)


def main():
    use_style()
    rng = np.random.default_rng(170)

    # ---------------------------------------------------------- (a) misspecified kernel
    w, beta = hk.power_law_weights(N_TRUE, EPS, TAU0, BETA_MIN)
    model = hk.SumExpHawkes1D(mu=RATE * (1 - N_TRUE), w=w, beta=beta)
    burn = 5.0 / BETA_MIN
    span = N_WIN * max(WINDOWS)
    ev = model.simulate(burn + span, rng)
    ev = ev[ev > burn] - burn
    n_hat = {}
    for W in WINDOWS:
        starts = np.arange(N_WIN) * (span / N_WIN)
        n_hat[W] = fit_windows(ev, W, starts)
    mean_a = np.array([n_hat[W][:, 0].mean() for W in WINDOWS])
    se_a = np.array([n_hat[W][:, 0].std(ddof=1) / np.sqrt(N_WIN) for W in WINDOWS])
    beta_hat = np.array([n_hat[W][:, 1].mean() for W in WINDOWS])

    # ---------------------------------------------------------- (b) seasonality
    def profile(t):
        x = (t % DAY) / DAY
        return 1.0 + 3.0 * (2 * x - 1) ** 2          # four times busier at the open and close

    base = RATE / 2.0                                 # mean of profile is 2
    raw, des = [], []
    days = []
    for d in range(N_DAYS):
        t = hk.simulate_inhomogeneous_poisson(lambda s: base * profile(s), DAY, base * 4.0, rng)
        days.append(t)
    # Seasonal profile estimated from all days: 30-minute bins of the pooled event counts.
    pooled = np.concatenate(days)
    edges = np.linspace(0, DAY, 14)
    counts, _ = np.histogram(pooled, edges)
    est_rate = counts / (N_DAYS * np.diff(edges))
    centres = 0.5 * (edges[1:] + edges[:-1])

    def est_profile(s):
        return np.interp(s, centres, est_rate)

    grid = np.linspace(0, DAY, 20001)
    for t in days:
        raw.append(hk.fit_exp_hawkes_1d(t, DAY)[1])
        tt, Tt = hk.time_change(t, est_profile, grid)
        des.append(hk.fit_exp_hawkes_1d(tt, Tt)[1])
    raw, des = np.array(raw), np.array(des)

    # ---------------------------------------------------------- (c) regime switches
    sw = []
    for d in range(N_DAYS):
        # rate alternates between 0.2 and 1.0 per second, mean regime life 600 s
        t0, t = 0.0, []
        hi = rng.random() < 0.5
        while t0 < DAY:
            life = min(rng.exponential(600.0), DAY - t0)
            r = 1.0 if hi else 0.2
            k = rng.poisson(r * life)
            t.append(t0 + np.sort(rng.uniform(0, life, k)))
            t0 += life
            hi = not hi
        sw.append(hk.fit_exp_hawkes_1d(np.concatenate(t), DAY)[1])
    sw = np.array(sw)

    # ---------------------------------------------------------- figure
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.2, 2.5))
    lag = np.geomspace(1e-2, 1e5, 300)
    a1.loglog(lag, model.kernel(lag), "-")
    a1.loglog(lag, N_TRUE * EPS * TAU0 ** EPS * (lag + TAU0) ** (-1 - EPS), "--", color="0.35")
    a1.set_xlabel(r"Lag $t$ (s)")
    a1.set_ylabel(r"Kernel $\phi(t)$ (s$^{-1}$)")
    a1.set_ylim(1e-12, 1e3)
    a1.plot([], [], "-", color="C0", label="Sum of 16 Exponentials")
    a1.plot([], [], "--", color="0.35", label=r"$\propto (t+\tau_0)^{-1.15}$")
    a1.legend(loc="upper right", fontsize=7)
    panel_label(a1, "(a)")

    a2.errorbar(WINDOWS, mean_a, yerr=2 * se_a, fmt="o-", ms=3.5, label="Exponential Fit")
    a2.axhline(N_TRUE, color="0.3", ls=":", label=r"True $n = 0.9$")
    a2.set_xscale("log")
    a2.set_xlabel(r"Fitting Window $W$ (s)")
    a2.set_ylabel(r"Branching Ratio $\hat n$")
    a2.set_ylim(0, 1.3)
    a2.legend(loc="upper left", fontsize=7)
    panel_label(a2, "(b)")

    cases = [("Seasonal,\nRaw", raw), ("Seasonal,\nTime-Changed", des), ("Regime\nSwitches", sw)]
    for k, (lab, v) in enumerate(cases):
        a3.plot(np.full(len(v), k) + rng.uniform(-0.12, 0.12, len(v)), v, "o", ms=2.5, mfc="white",
                color=f"C{k}")
        a3.errorbar(k, v.mean(), yerr=2 * v.std(ddof=1) / np.sqrt(len(v)), fmt="D", ms=4, color="k")
    a3.axhline(0.0, color="0.3", ls=":")
    a3.set_xticks(range(3))
    a3.set_xticklabels([c[0] for c in cases], fontsize=7.5)
    a3.minorticks_off()
    a3.tick_params(axis="y", which="minor", right=True)
    a3.set_xlim(-0.6, 2.6)
    a3.set_xlabel("Poisson Data, True $n = 0$")
    a3.set_ylabel(r"Branching Ratio $\hat n$")
    a3.set_ylim(-0.1, 1.2)
    panel_label(a3, "(c)")
    fig.tight_layout()
    save(fig, "fig17_hawkes_pitfalls", FIG, width_in=7.2)

    def pm(v):
        return fmt_pm(v.mean(), v.std(ddof=1) / np.sqrt(len(v)), 3)

    write_numbers("e17_hawkes_pitfalls", {
        "EseventeenNevents": f"{len(ev):,}".replace(",", "{,}"),
        "EseventeenShort": fmt_pm(mean_a[0], se_a[0], 2),
        "EseventeenLong": fmt_pm(mean_a[-1], se_a[-1], 2),
        "EseventeenSeasonRaw": pm(raw),
        "EseventeenSeasonDes": pm(des),
        "EseventeenSwitch": pm(sw),
        "EseventeenDays": str(N_DAYS),
        "EseventeenBetaInv": fmt(1.0 / beta_hat.mean(), 2),
    })


if __name__ == "__main__":
    main()
