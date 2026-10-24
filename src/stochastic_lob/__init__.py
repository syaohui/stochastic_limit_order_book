"""Stochastic models of the limit order book, and the liquidity provider's problems.

Modules
-------
markov_lob         Markovian book of Cont, Stoikov and Talreja (2010); zero intelligence;
                   maximum-likelihood rate estimation from the event stream
best_queues        reduced-form best-queue model of Cont and de Larrard (2013)
queue_reactive     queue-reactive model of Huang, Lehalle and Rosenbaum (2015): Model I,
                   intensity estimation, and a reduced Model III with a reference price
hawkes             Hawkes processes: exponential kernels (simulation, MLE, diagnostics),
                   slowly decaying kernels, seasonality, the price model of Bacry et al. (2013)
impact             propagator model and its estimation; order flow imbalance
                   (single, multi-level, integrated)
latent_book        latent order book and metaorder impact, from linear to square root
microprice         the micro-price of Stoikov (2018)
market_making      Avellaneda-Stoikov, GLFT closed form, exact GLFT with inventory limits
                   and drift, signal-aware quoting, fill-intensity estimation
adverse_selection  the Glosten-Milgrom (1985) model
queue_position     fill probability and value of a resting order by queue position
markouts           markouts and realized spread of passive fills
amm                loss-versus-rebalancing for constant-product and concentrated liquidity
lobster            read and write order book data in the LOBSTER format
"""
from . import (adverse_selection, amm, best_queues, hawkes, impact, latent_book, lobster,
               market_making, markouts, markov_lob, microprice, queue_position, queue_reactive)

__all__ = ["adverse_selection", "amm", "best_queues", "hawkes", "impact", "latent_book",
           "lobster", "market_making", "markouts", "markov_lob", "microprice",
           "queue_position", "queue_reactive"]
__version__ = "0.3.0"
