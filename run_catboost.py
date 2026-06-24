#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_catboost.py — CatBoost calibration, nirs4all-idiomatic.

Preprocessing search (shape x scatter x phase-2) via ``_or_`` steps; the gradient
boosting model uses a fixed number of trees (no hyperparameter search, as in the
benchmark). CatBoost runs on GPU.

Example
-------
    python run_catboost.py --data-root ../../Data/regression --output-dir ./results_catboost
"""

from __future__ import annotations

import argparse
from pathlib import Path

from catboost import CatBoostRegressor

from benchmark_common import (
    SEED, cv_splitter, nonlinear_preprocessing_search, run_over_datasets,
)


def build_pipeline(n_estimators: int = 500, device: str = "GPU") -> list:
    return [
        *nonlinear_preprocessing_search(),
        cv_splitter(),
        {
            "model": CatBoostRegressor(
                iterations=int(n_estimators),
                loss_function="RMSE",
                task_type=device,          # "GPU" or "CPU"
                devices="0",
                random_seed=SEED,
                verbose=False,
                allow_writing_files=False,
            ),
            "name": "CatBoost",
        },
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="./results_catboost")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()
    run_over_datasets(lambda: build_pipeline(args.n_estimators, args.device),
                      "CatBoost", Path(args.data_root), Path(args.output_dir),
                      datasets=args.datasets, limit=args.limit, verbose=args.verbose)


if __name__ == "__main__":
    main()
