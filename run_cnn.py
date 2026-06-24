#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_cnn.py — 1D-CNN (NICON) calibration, nirs4all-idiomatic.

Uses nirs4all's native NICON convolutional architecture (``customizable_nicon``).
Preprocessing search (shape x scatter x phase-2) via ``_or_`` steps; a small
architecture search is performed by SPXY-grouped CV via nirs4all's finetuning,
then the best model is retrained for more epochs and evaluated on the test set.

Note
----
This uses the native (TensorFlow) NICON shipped with nirs4all. The original study
used a PyTorch re-implementation; the two share the same architecture family but
may differ slightly numerically.

Example
-------
    python run_cnn.py --data-root ../../Data/regression --output-dir ./results_cnn
"""

from __future__ import annotations

import argparse
import os
import sys

# TensorFlow must be told to hide the GPU *before* it is imported. Pass --cpu (or
# set N4A_CNN_CPU=1) to run NICON on CPU -- useful where the TF build lacks GPU
# kernels for very recent architectures (e.g. CUDA_ERROR_INVALID_HANDLE on Blackwell).
if "--cpu" in sys.argv or os.environ.get("N4A_CNN_CPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path

from nirs4all.operators.models.tensorflow.nicon import customizable_nicon

from benchmark_common import (
    SEED, conv_preprocessing_search, cv_splitter, run_over_datasets,
)


def build_pipeline(n_trials: int = 30, epochs_search: int = 50, epochs_final: int = 500) -> list:
    return [
        *conv_preprocessing_search(),
        cv_splitter(),
        {
            "model": customizable_nicon,
            "name": "CNN-1D",
            "finetune_params": {
                "n_trials": int(n_trials),
                "sampler": "tpe",
                "seed": SEED,
                "metric": "rmse",
                "model_params": {
                    "filters_1": [8, 16, 32, 64],
                    "filters_3": [8, 16, 32, 64],
                },
                "train_params": {"epochs": int(epochs_search), "verbose": 0},
            },
            "train_params": {"epochs": int(epochs_final), "verbose": 0},
        },
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="./results_cnn")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--epochs-search", type=int, default=50)
    p.add_argument("--epochs-final", type=int, default=500)
    p.add_argument("--cpu", action="store_true", help="Run NICON on CPU (hides the GPU from TensorFlow).")
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()
    run_over_datasets(
        lambda: build_pipeline(args.n_trials, args.epochs_search, args.epochs_final),
        "CNN-1D", Path(args.data_root), Path(args.output_dir),
        datasets=args.datasets, limit=args.limit, verbose=args.verbose)


if __name__ == "__main__":
    main()
