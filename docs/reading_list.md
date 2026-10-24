# Reading list

A map of the literature behind the slides, grouped by question. Each entry says what the work contributes. Full bibliographic entries are in [`slides/refs.bib`](../slides/refs.bib). Preprints are marked with their arXiv number. All entries were checked against publisher, arXiv, or RePEc records in October 2026.

## Start here

| Work | Why read it |
|---|---|
| Gould, Porter, Williams, McDonald, Fenn and Howison (2013), *Limit order books*, Quantitative Finance | Survey of empirical facts and models up to 2013. |
| Bouchaud, Bonart, Donier and Gould (2018), *Trades, Quotes and Prices*, Cambridge University Press | Book-length treatment of order flow, impact, and the latent order book. |
| Cartea, Jaimungal and Penalva (2015), *Algorithmic and High-Frequency Trading*, Cambridge University Press | Stochastic control for market making, execution, and signals. |
| Abergel, Anane, Chakraborti, Jedidi and Muni Toke (2016), *Limit Order Books*, Cambridge University Press | Markovian and Hawkes order book models with proofs. |

## Queueing models of the book

| Work | Contribution |
|---|---|
| Bouchaud, Mézard and Potters (2002), Quantitative Finance | Power-law placement of limit orders and the hump-shaped average book (Paris Bourse). |
| Smith, Farmer, Gillemot and Krishnamurthy (2003), Quantitative Finance | Zero-intelligence double auction; dimensional analysis of spread and depth. |
| Farmer, Patelli and Zovko (2005), PNAS | The zero-intelligence model explains most of the cross-sectional variation in mean spread on the LSE. |
| Cont, Stoikov and Talreja (2010), Operations Research | Markovian book with power-law placement; Laplace-transform formulas for price moves. |
| Cont and de Larrard (2013), SIAM J. Financial Mathematics | Best-queue model: direction of the next move, infinite mean duration, diffusion limit. |
| Huang, Lehalle and Rosenbaum (2015), JASA | Queue-reactive model: all rates depend on queue size. |
| Robert and Rosenbaum (2011), J. Financial Econometrics | Uncertainty zones; the mean reversion ratio of price changes. |
| Wu, Rambaldi, Muzy and Bacry (2019), arXiv:1901.08938 | Queue-reactive Hawkes models. |

## Self-exciting order flow

| Work | Contribution |
|---|---|
| Hawkes (1971), Biometrika | The process and its spectrum. |
| Bacry, Delattre, Hoffmann and Muzy (2013), Quantitative Finance | Hawkes price model; signature plot from cross-excitation. |
| Hardiman, Bercot and Bouchaud (2013), EPJ B | Order flow close to criticality. |
| Filimonov and Sornette (2015), Quantitative Finance | Regime shifts and outliers bias the branching ratio upward. |
| Bacry, Mastromatteo and Muzy (2015), Market Microstructure and Liquidity | Review of Hawkes processes in finance. |
| Bacry, Jaisson and Muzy (2016), Quantitative Finance | Nonparametric estimation of slowly decaying kernels; eight-dimensional order book model. |
| Jaisson and Rosenbaum (2015), Annals of Applied Probability | Limits of nearly unstable Hawkes processes. |
| El Euch, Fukasawa and Rosenbaum (2018), Finance and Stochastics | Rough volatility and leverage from microstructure. |
| Wang, Wu and Zhu (2026), arXiv:2608.07709 | Rough Hawkes-Heston model with common price and volatility jumps. |

## Order flow and price impact

| Work | Contribution |
|---|---|
| Lillo and Farmer (2004), SNDE | Long memory of order signs. |
| Bouchaud, Gefen, Potters and Wyart (2004), Quantitative Finance | Propagator model; critical decay that keeps prices diffusive. |
| Almgren, Thum, Hauptmann and Li (2005), Risk | Direct estimation of equity market impact; a 3/5 power law. |
| Eisler, Bouchaud and Kockelkoren, arXiv:0904.0900 | The impact of limit orders and cancellations, not only trades. |
| Gatheral (2010), Quantitative Finance | No-dynamic-arbitrage constraints on impact shape and decay. |
| Tóth et al. (2011), Physical Review X | Square-root impact and the critical nature of liquidity. |
| Donier, Bonart, Mastromatteo and Bouchaud (2015), Quantitative Finance | Latent order book model for non-linear impact. |
| Cont, Kukanov and Stoikov (2014), J. Financial Econometrics | Order flow imbalance and linear short-horizon impact. |
| Xu, Gould and Howison (2019), arXiv:1907.06230 | Multi-level OFI. |
| Cont, Cucuringu and Zhang (2023), Quantitative Finance | Integrated OFI; cross-impact is contemporaneously redundant but forecasts at short lags. |
| Stoikov (2018), Quantitative Finance | The micro-price. |
| Lehalle and Neuman (2019), Finance and Stochastics | Optimal trading with a Markov signal; imbalance predicts and mean-reverts. |
| Sirignano and Cont (2019), Quantitative Finance | Deep learning finds a universal price formation mechanism. |
| Zhang, Zohren and Roberts (2019), arXiv:1808.03668 | DeepLOB. |
| Kolm, Turiel and Westray (2023), Mathematical Finance | Deep OFI features at multiple horizons. |

## Liquidity provision

| Work | Contribution |
|---|---|
| Glosten and Milgrom (1985), JFE | Adverse selection sets the spread. |
| Avellaneda and Stoikov (2008), Quantitative Finance | Reservation price and inventory-skewed quotes. |
| Guéant, Lehalle and Fernandez-Tapia (2013), Mathematics and Financial Economics | Exact solution with inventory limits through a linear ODE system. |
| Guéant (2017), Applied Mathematical Finance | General intensities and multi-asset market making. |
| Moallemi and Yuan (2017), working paper | Value of queue position. |
| Cartea and Sánchez-Betancourt (2025), SIAM J. Financial Mathematics | Brokers, informed traders, and signals from toxic flow. |
| Cartea, Duran-Martin and Sánchez-Betancourt (2023), arXiv:2312.05827 | Predicting toxic client flow online. |
| Gašperov, Begušić, Posedel Šimović and Kostanjčar (2021), Mathematics | Survey of reinforcement learning for market making. |
| Vischi, Stradi and Marchesi (2026), arXiv:2610.09691 | Optimal regret for online market making with order book feedback. |
| Maran and Restelli (2026), COLT, arXiv:2605.19584 | The value of observing the order book in online market making. |

## Latency and market design

| Work | Contribution |
|---|---|
| Budish, Cramton and Shim (2015), QJE | The speed race and frequent batch auctions. |
| Aquilina, Budish and O'Neill (2022), QJE | Measuring latency races on LSE message data. |

## Automated market makers

| Work | Contribution |
|---|---|
| Milionis, Moallemi, Roughgarden and Zhang (2022), arXiv:2208.06046 | Loss-versus-rebalancing. |
| Milionis, Moallemi and Roughgarden (2023), arXiv:2305.14604 | Arbitrage profits with fees and discrete blocks. |

## Generative and foundation models of order flow

| Work | Contribution |
|---|---|
| Nagy et al. (2023), arXiv:2309.00638 | Token-level autoregressive model of message flow. |
| Nagy et al. (2025), ICML, LOB-Bench | Benchmark for generated LOB data, including impact metrics. |
| Linna, Baltakys, Iosifidis and Kanniainen (2025), arXiv:2511.12563 | LOBERT, an encoder-only foundation model for messages. |
| Kawawa-Beaudan, Sood, Papasotiriou, Borrajo and Veloso (2026), arXiv:2602.23784 | TradeFM, a cross-asset generative model of trade flow. |
| Mohl et al. (2026), ICAIF, arXiv:2610.28753 | Interpreting a state-space foundation model of the book. |
