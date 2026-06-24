#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_pls.py — PLS calibration, nirs4all-idiomatic.

Preprocessing search (shape x scatter x representation x scaler) is expressed as
``_or_`` steps; the number of PLS latent variables is selected by SPXY-grouped CV
via nirs4all's native finetuning. Best configuration is refit on the full
calibration set and evaluated on the external test set.

Example
-------
    python run_pls.py --data-root ../../Data/regression --output-dir ./results_pls
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.cross_decomposition import PLSRegression

from benchmark_common import (
    SEED, cv_splitter, linear_preprocessing_search, run_over_datasets,
)


def build_pipeline() -> list:
    return [
        *linear_preprocessing_search(),
        cv_splitter(),
        {
            "model": PLSRegression(scale=False),
            "name": "PLS",
            "finetune_params": {
                "n_trials": 30,
                "sampler": "grid",          # exhaustive over n_components
                "approach": "grouped",
                "seed": SEED,
                "metric": "rmse",
                "model_params": {"n_components": ("int", 1, 30)},
            },
        },
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, help="Folder containing the dataset subfolders.")
    p.add_argument("--output-dir", default="./results_pls")
    p.add_argument("--datasets", nargs="*", default=None, help="Optional subset of dataset folder names.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()
    run_over_datasets(build_pipeline, "PLS", Path(args.data_root), Path(args.output_dir),
                      datasets=args.datasets, limit=args.limit, verbose=args.verbose)


if __name__ == "__main__":
    main()
