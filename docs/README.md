# Documentation

This folder explains the models, the statistics and the code behind the slides. Each topic page follows the same order as its deck: the question, the model, what it predicts, how to estimate it, what this repository's experiments show, what published research reports, and what it means for a liquidity provider.

## Topic pages

| # | Topic | Deck | Page |
|---|---|---|---|
| 1 | Zero intelligence and the Markovian book | [`01-markovian-order-book.pdf`](../slides/decks/01-markovian-order-book.pdf) | [topics/01-markovian-order-book.md](topics/01-markovian-order-book.md) |
| 2 | The best queues (Cont-de Larrard) | [`02-best-queues.pdf`](../slides/decks/02-best-queues.pdf) | [topics/02-best-queues.md](topics/02-best-queues.md) |
| 3 | Queue-reactive models | [`03-queue-reactive-models.pdf`](../slides/decks/03-queue-reactive-models.pdf) | [topics/03-queue-reactive.md](topics/03-queue-reactive.md) |
| 4 | Self-exciting order flow (Hawkes) | [`04-hawkes-order-flow.pdf`](../slides/decks/04-hawkes-order-flow.pdf) | [topics/04-hawkes.md](topics/04-hawkes.md) |
| 5 | Price impact, propagators, latent book | [`05-price-impact.pdf`](../slides/decks/05-price-impact.pdf) | [topics/05-price-impact.md](topics/05-price-impact.md) |
| 6 | Order flow imbalance and the micro-price | [`06-order-flow-imbalance-microprice.pdf`](../slides/decks/06-order-flow-imbalance-microprice.pdf) | [topics/06-order-flow.md](topics/06-order-flow.md) |
| 7 | Optimal market making | [`07-optimal-market-making.pdf`](../slides/decks/07-optimal-market-making.pdf) | [topics/07-market-making.md](topics/07-market-making.md) |
| 8 | Adverse selection and queue priority | [`08-adverse-selection-queue-priority.pdf`](../slides/decks/08-adverse-selection-queue-priority.pdf) | [topics/08-adverse-selection.md](topics/08-adverse-selection.md) |
| 9 | Liquidity in automated market makers | [`09-amm-liquidity.pdf`](../slides/decks/09-amm-liquidity.pdf) | [topics/09-amm.md](topics/09-amm.md) |

The overview deck, [`slides/stochastic-lob-slides.pdf`](../slides/stochastic-lob-slides.pdf), covers all of these in one pass, plus recent research (2023 to 2026) and open problems.

## Using the code

- [api.md](api.md): every module and its main functions, with units.
- [data.md](data.md): running the estimators on real order book data in the LOBSTER format.
- [reproducibility.md](reproducibility.md): how numbers flow from code to slides, seeds, run times, and how to add an experiment.
- [reading_list.md](reading_list.md): the literature, grouped by question.

## Conventions

- Time in seconds, prices in ticks, sizes in orders (unit-size) unless a page says otherwise. Market-making examples use ticks or the units of Avellaneda and Stoikov (2008) as stated.
- Every symbol is defined where it first appears on a page, with its unit.
- Every simulated number carries a standard error and a sample size. Numbers quoted from papers are as reported there; this repository contains no market data.
- Where a result is derived here rather than taken from a paper, the page says so.
