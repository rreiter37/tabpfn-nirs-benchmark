#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_ridge.py — Ridge calibration, nirs4all-idiomatic.

Same preprocessing search as PLS; the Ridge regularization strength ``alpha`` is
tuned on a log scale by SPXY-grouped CV via nirs4all's native (Optuna) finetuning.

Example
-------
    python run_ridge.py --data-root ../../Data/regression --output-dir ./results_ridge
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.linear_model import Ridge

from benchmark_common import (
    SEED, cv_splitter, linear_preprocessing_search, run_over_datasets,
)


def build_pipeline() -> list:
    return [
        *linear_preprocessing_search(),
        cv_splitter(),
        {
            "model": Ridge(random_state=SEED),
            "name": "Ridge",
            "finetune_params": {
                "n_trials": 30,
                "sampler": "tpe",
                "approach": "grouped",
                "seed": SEED,
                "metric": "rmse",
                "model_params": {
                    "alpha": {"type": "float", "min": 1e-3, "max": 100.0, "log": True},
                },
            },
        },
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="./results_ridge")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()
    run_over_datasets(build_pipeline, "Ridge", Path(args.data_root), Path(args.output_dir),
                      datasets=args.datasets, limit=args.limit, verbose=args.verbose)


if __name__ == "__main__":
    main()
