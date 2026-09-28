# -*- coding: utf-8 -*-
"""
run_catboost.py — CatBoost arm of the benchmark (CatBoost-pp).

CatBoost is run at fixed hyperparameters, as in the manuscript: only the
preprocessing is searched, over the two-phase space (21 + 9 = 30 configurations)
scored by grouped 3-fold SPXY cross-validation.

The tree count is the one setting that differs between the two stages:

    iterations = 200  while scoring preprocessing configurations
    iterations = 500  for the final refit and the reported predictions

Growing 500 trees for every cross-validated configuration, as the first version
of this repository did, multiplies the search cost by 2.5 without changing which
preprocessing wins.

The Supplementary Material reports a separate study where CatBoost is also given
a hyperparameter search; that arm is not part of this repository, which
reproduces the main manuscript.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from benchmark_common import (
    SEED, SearchConfig, load_dataset, run_over_datasets, score_of,
    two_phase_search, write_outputs,
)

ITERATIONS_SEARCH = 200
ITERATIONS_FINAL = 500


def make_model(task: str, iterations: int, device: str):
    """Build a CatBoost estimator; ``device`` is CatBoost's ``task_type``."""
    common = dict(iterations=int(iterations), random_seed=SEED, task_type=device,
                  verbose=False, allow_writing_files=False)
    if device == "GPU":
        common["devices"] = "0"
    if task == "regression":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(loss_function="RMSE", **common)
    from catboost import CatBoostClassifier
    return CatBoostClassifier(loss_function="Logloss", **common)


def make_evaluator(task: str, X: np.ndarray, y: np.ndarray, device: str):
    def evaluate(cfg: SearchConfig, folds: List[Tuple[np.ndarray, np.ndarray]]):
        scores = []
        for train_idx, valid_idx in folds:
            pipe = Pipeline(cfg.steps() + [
                ("model", make_model(task, ITERATIONS_SEARCH, device))])
            pipe.fit(X[train_idx], y[train_idx])
            scores.append(score_of(task, y[valid_idx], pipe.predict(X[valid_idx])))
        return float(np.mean(scores)), {"iterations": ITERATIONS_SEARCH}

    return evaluate


def run_one(folder: Path, output_dir: Path, *, task: str, device: str,
            verbose: int) -> Dict[str, Any]:
    X, y, X_test, y_test = load_dataset(folder)

    encoder = None
    if task == "classification":
        encoder = LabelEncoder().fit(np.concatenate([y, y_test]) if y_test is not None else y)
        y = encoder.transform(y)

    best_config, best_params, trace = two_phase_search(
        X, y, make_evaluator(task, X, y, device), linear=False, verbose=verbose)

    final = Pipeline(best_config.steps() + [
        ("model", make_model(task, ITERATIONS_FINAL, device))])
    final.fit(X, y)
    y_pred = np.asarray(final.predict(X_test)).ravel()
    if encoder is not None:
        y_pred = encoder.inverse_transform(np.asarray(y_pred, dtype=int))

    return write_outputs(output_dir, folder.name, best_config,
                         {**best_params, "iterations_final": ITERATIONS_FINAL},
                         trace, y_pred, y_test, task)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="regression", choices=["regression", "classification"])
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()

    out = args.output_dir or f"./results_catboost_{args.task}"
    run_over_datasets(
        lambda folder, outdir: run_one(folder, outdir, task=args.task,
                                       device=args.device, verbose=args.verbose),
        "CatBoost-pp", Path(args.data_root), Path(out),
        datasets=args.datasets, limit=args.limit, verbose=args.verbose)


if __name__ == "__main__":
    main()
