# Running the estimators on real order book data

This repository contains no market data. Every figure comes from simulation. The estimators, though, are written for real data, and [`examples/lobster_pipeline.py`](../examples/lobster_pipeline.py) runs them on one day of order book data in the LOBSTER format.

## The LOBSTER format

[LOBSTER](https://lobsterdata.com) reconstructs NASDAQ order books. One stock-day comes as two CSV files with one row per event.

**Message file**: `time, type, order_id, size, price, direction`

| Column | Meaning |
|---|---|
| time | seconds after midnight |
| type | 1 new limit order, 2 partial cancellation, 3 full deletion, 4 visible execution, 5 hidden execution, 6 cross trade, 7 trading halt |
| size | shares |
| price | dollars times 10,000 |
| direction | +1 buy limit order, -1 sell limit order; for executions, the side of the resting order (so -1 is a buyer-initiated trade) |

**Orderbook file**: `ask price 1, ask size 1, bid price 1, bid size 1, ask price 2, ...`, the book after the event on the same row. Empty levels carry the dummy prices 9999999999 (ask) and -9999999999 (bid) with size 0.

LOBSTER publishes free sample files (for example `AMZN_2012-06-21_34200000_57600000_message_10.csv` and its orderbook file). Download them from the LOBSTER website yourself and check its terms of use.

## Run the pipeline

```bash
python examples/lobster_pipeline.py MESSAGE.csv ORDERBOOK.csv --tick 0.01 --lot 100 --out report
```

```bash
python examples/lobster_pipeline.py --simulate --out report
```

The second command writes a simulated day in LOBSTER format (into `report/simulated_lobster/`) and runs the same analysis, so you can check the pipeline before you have data. It writes `report/summary.json` and a figure.

| Output | Method | Topic |
|---|---|---|
| `rates` | Poisson MLE of $\lambda(i)$, $\theta(i)$, $\mu$ | 1 |
| `p_up` | conditional frequency of an up move with clustered errors; constant-rate (Cont-de Larrard) fit | 2 |
| `hawkes` | bivariate exponential Hawkes fit of market buys and sells, KS test of residuals | 4 |
| `ofi` | OFI and five-level OFI regressions, 10 s windows, out-of-sample $R^2$ | 6 |
| `microprice` | micro-price adjustment by imbalance and spread | 6 |
| `markouts` | mean markout of passive fills at 0, 1, 5, 30, 120 s | 8 |

## In Python

```python
from stochastic_lob import lobster, markov_lob as ml
d = lobster.read("MSG.csv", "BOOK.csv")
rec = lobster.to_record(d, tick=0.01, lot=100, K=40)     # same LOBRecord as the simulator
est = ml.estimate_rates(rec, L=10)
```

## Caveats for real data

- **Sizes.** Orders are converted to lots (`--lot`, default 100 shares) and rounded. Queue models assume unit orders.
- **Hidden executions** (type 5) are counted as market orders, although they do not deplete the visible queue. Filter them if that matters for your question.
- **Visible depth.** A 10-level file shows only the first ten nonempty prices per side. Cancellation rates far from the touch are biased (topic 1).
- **Seasonality.** Intraday patterns inflate the Hawkes branching ratio (topic 4). Use `--deseason` for a time change by a 30-minute rate profile, or better, a profile estimated over many days.
- **Time stamps.** Simultaneous events (same time stamp) are dropped before the Hawkes fit.
- **Overlapping horizons.** Markout standard errors treat fills as independent; for long horizons use blocks of the day.
- **One day is one sample.** Rates and kernels change across days; fit several days and report the spread.
