#!/usr/bin/env python
"""
Run the full `neurocopula` benchmark suite and write the results.

This is the script behind the numbers in the README. It runs every case in
`neurocopula.benchmark.STANDARD_SUITE` across several seeds -- several,
because a single flow fit varies enough with initialisation that a one-seed
difference between models is not evidence of anything.

Outputs, all under ``benchmarks/results/``:

- ``raw_results.csv``   -- one row per (case, model, seed)
- ``summary.csv``       -- mean and standard deviation per (case, model)
- ``tail_table.csv``    -- estimated vs theoretical tail dependence

Usage::

    python benchmarks/run_benchmark.py                # 3 seeds, full suite
    python benchmarks/run_benchmark.py --seeds 5      # more seeds, tighter error bars
    python benchmarks/run_benchmark.py --quick        # 1 seed, fewer epochs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from neurocopula.benchmark import (
    STANDARD_SUITE,
    run_benchmark,
    summarize,
    tail_dependence_table,
)

RESULTS = Path(__file__).parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds (default 3)")
    ap.add_argument("--epochs", type=int, default=600, help="max training epochs per fit")
    ap.add_argument("--n-mc", type=int, default=100_000, help="Monte Carlo draws per model")
    ap.add_argument("--quick", action="store_true", help="1 seed, 200 epochs, 20k draws")
    args = ap.parse_args()

    if args.quick:
        args.seeds, args.epochs, args.n_mc = 1, 200, 20_000

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 50)

    results = run_benchmark(
        STANDARD_SUITE,
        seeds=tuple(range(args.seeds)),
        epochs=args.epochs,
        n_mc=args.n_mc,
        verbose=True,
    )
    results.to_csv(RESULTS / "raw_results.csv", index=False)

    summary = summarize(results)
    summary.to_csv(RESULTS / "summary.csv")
    tails = tail_dependence_table(results)
    tails.to_csv(RESULTS / "tail_table.csv", index=False)

    print("\n=== held-out copula log-likelihood (higher is better) ===")
    print(results.pivot_table(index="case", columns="model",
                              values="test_copula_loglik", aggfunc="mean").round(4))
    print("\n=== tail dependence vs theory ===")
    print(tails.to_string(index=False))
    print("\n=== fit time, seconds ===")
    print(results.pivot_table(index="case", columns="model",
                              values="fit_seconds", aggfunc="mean").round(2))
    print(f"\nwrote {RESULTS}/raw_results.csv, summary.csv, tail_table.csv")


if __name__ == "__main__":
    main()
