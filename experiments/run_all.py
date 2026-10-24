"""Run every experiment in order. Takes about fifteen minutes on a laptop."""
import importlib
import time

EXPERIMENTS = [
    "e1_markov_book",
    "e2_best_queues",
    "e3_queue_reactive",
    "e4_hawkes",
    "e5_signature",
    "e6_propagator",
    "e7_market_making",
    "e8_ofi",
    "e9_queue_value",
    "e10_markouts",
    "e11_glft_exact",
    "e12_signal",
    "e13_lvr",
    "e14_mlofi",
    "e15_rate_estimation",
    "e16_queue_reactive_price",
    "e17_hawkes_pitfalls",
    "e18_propagator_estimation",
    "e19_latent_book",
    "e20_microprice",
    "e21_fill_estimation",
    "e22_glosten_milgrom",
    "e23_cdl_on_book",
]

if __name__ == "__main__":
    for name in EXPERIMENTS:
        t0 = time.time()
        importlib.import_module(name).main()
        print(f"{name:20s} {time.time() - t0:6.1f} s")
