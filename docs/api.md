# API reference

The package `stochastic_lob` needs only NumPy, SciPy and Matplotlib. Install it from the repository root with `pip install -e .`. Units follow the docstrings: seconds, ticks and unit orders unless stated. Every function that draws random numbers takes `rng` (a seed or a `numpy.random.Generator`).

## Order book models

### `markov_lob`: zero intelligence and the Cont-Stoikov-Talreja book (topic 1)

| Function | What it does |
|---|---|
| `power_law_params(L, lam1, a, theta0, mu, tick)` | rates $\lambda(i) = \lambda_1 i^{-a}$, constant $\theta$ |
| `zero_intelligence_params(L, lam0, theta0, mu, tick)` | constant rates over $L$ ticks |
| `simulate_markov_lob(params, n_events, rng, init_depth, burn_in, record_levels)` | exact Gillespie simulation; returns a `LOBRecord` |
| `estimate_rates(rec, L)` | Poisson MLE of $\lambda(i)$, $\theta(i)$, $\mu$ with standard errors (needs `record_levels > 0`) |
| `empirical_p_up(rec, max_q)` | frequency of an up move given $(q_b, q_a)$, with errors clustered by price-move spell |

`LOBRecord` holds, per event: time `t` (s), `best_bid`, `best_ask` (ticks), `q_bid`, `q_ask` (orders), `event` (code 0 to 5, see `EVENT_NAMES`), `event_price` (ticks), and optionally `bid_depth`, `ask_depth` of shape (events, K) on the tick grid from each best quote.

### `best_queues`: Cont and de Larrard (topic 2)

`first_passage_density`, `survival`, `p_up(q_b, q_a, lam, nu)`, `p_up_balanced`, `p_up_arctan`, `p_up_linear_system`, `duration_survival`, `duration_tail_balanced`, `diffusion_coefficient_balanced`, `simulate_durations`.

### `queue_reactive`: Huang, Lehalle and Rosenbaum (topic 3)

| Function | What it does |
|---|---|
| `QueueRates(limit, cancel, market)` | intensity functions of queue size (1/s) |
| `stationary_distribution(rates, q_max)` | Model I product formula |
| `simulate_queue`, `simulate_queue_events` | exact simulation of one queue; the second records every event |
| `estimate_intensities(wait, q_before, kind, n_max, censored)` | nonparametric MLE with standard errors |
| `hlr_like_rates()` | illustrative intensities with HLR's empirical shapes |
| `Model3Params`, `simulate_model3(p, t_end, rng)` | reduced Model III with a reference price |
| `mean_reversion_ratio(signs)`, `realized_vol(...)` | $\eta = N_c/(2N_a)$ and volatility over a horizon |

### `queue_position`: fill probability and value of a resting order (topic 8)

`QueueRates(lam, mu, theta, nu, jump)`, `p_up_state_dependent(r, b_max, c_max)` (exact $p_\uparrow$ when depletion is $\mu + \theta q$), `solve_queue_value(r, a_max)` (absorbing Markov chain), `simulate_order` (Monte Carlo check).

## Order flow and impact

### `hawkes`: self-exciting order flow (topic 4)

| Function | What it does |
|---|---|
| `ExpHawkes(mu, alpha, beta)` | multivariate exponential Hawkes: `simulate`, `log_likelihood`, `time_rescaled_residuals`, `stationary_intensity` |
| `fit_exp_hawkes(times, types, T, D)` | MLE with observed-information standard errors |
| `fit_exp_hawkes_1d(times, T)` | univariate fit; returns $\mu$, $\hat n$, $\hat\beta$, s.e. of $\hat n$ |
| `SumExpHawkes1D`, `power_law_weights(n, eps, tau0, beta_min)` | slowly decaying kernels as sums of exponentials |
| `simulate_inhomogeneous_poisson`, `time_change` | seasonality and its removal |
| `signature_plot_theory`, `realized_signature` | Hawkes price model of Bacry et al. (2013) |

### `impact`: propagators and order flow imbalance (topics 5 and 6)

Propagator: `long_memory_signs(n, gamma_eps)`, `propagator_price(signs, Gamma0, zeta)`, `response_function`, `diffusion_signature`, `critical_zeta`, `estimate_propagator(signs, dprice, max_lag)`, `block_bootstrap`.
OFI: `ofi_contributions`, `aggregate_ofi`, `multilevel_ofi_contributions`, `aggregate_multilevel_ofi`, `integrated_ofi`, `ols`, `ols_multi`, `out_of_sample_r2`.

### `latent_book`: metaorder impact from linear to square root (topic 5)

`self_similar_amplitude(u)` ($F(u)$, closed form), `impact_peak(Q, T, L, D)`, `linear_response_path`, `local_exponent`, `simulate_metaorder(m, T, L, D, t_end)` (finite-difference check).

### `microprice`: Stoikov's micro-price (topic 6)

`states(q_bid, q_ask, spread, n_imb, max_spread)`, `fit(idx, mid)` (returns `G`, the $g_i$, `B`), `predictors` (mid, weighted mid, micro-price), `mirror`.

## Liquidity provision

### `market_making` (topic 7)

| Function | What it does |
|---|---|
| `MMParams(S0, T, sigma, dt, gamma, k, A, drift)` | model parameters |
| `as_quotes`, `glft_quotes`, `symmetric_quotes` | Avellaneda-Stoikov, GLFT closed form, benchmark |
| `glft_matrix`, `glft_exact_quotes(Q, p, taus, drift)`, `glft_stationary_quotes` | exact solution with inventory limits and drift |
| `simulate(strategy, p, n_paths)`, `glft_strategy`, `certainty_equivalent` | backtest a quoting rule |
| `simulate_regime_signal(p, Q, m0, switch_rate, use_signal, shrink)` | quoting with a short-lived drift signal |
| `simulate_quote_spells`, `fit_fill_intensity`, `fit_fill_ratio_naive` | estimating $Ae^{-kd}$ with and without censoring |

### `adverse_selection`: Glosten-Milgrom (topic 8)

`quotes(pi, alpha)`, `simulate(alpha, n_trades, n_paths)`, `markouts_vs_value(sim)`.

### `markouts` (topic 8)

`passive_fills(rec)`, `mid_at(rec, times)`, `markout_matrix(rec, horizons)`, `mean_with_se`.

### `amm`: loss-versus-rebalancing (topic 9)

`cpmm_value`, `lvr_rate_cpmm`, `expected_lvr_cpmm`, `lvr_per_capital_concentrated(sigma, w)`, `simulate_lvr_cpmm`, `simulate_cpmm_arbitrage(sigma, T, block_dt, fee, n_paths)`, `range_position`, `simulate_lvr_range`.

## Data and figures

- `lobster`: `read(message, orderbook)`, `to_record(d, tick, lot, K)`, `write(rec, folder)`, `event_codes`. See [data.md](data.md).
- `plotting`: `use_style()` (journal style: inward ticks on four sides, print-size type), `panel_label`, `save(fig, stem, outdir)` (PNG, 300 dpi).
