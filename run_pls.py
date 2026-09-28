# -*- coding: utf-8 -*-
"""
run_pls.py — chemometric reference of the benchmark.

    --task regression      PLS-pp-hpo,    the reference every relative metric
                           (iRMSEP) is computed against.
    --task classification  PLS-DA-pp-hpo, the reference of the classification
                           benchmark.

The linear models search a wider preprocessing space than the tabular ones:
8 shapes instead of 7 -- they also try a Gaussian first derivative -- and their
phase 2 is a representation x scaler grid rather than a single operator. That
gives 24 + 3 x 12 = 60 configurations against 30, which the manuscript justifies
by the sensitivity of linear calibrations to these choices. Sharing one
preprocessing space between the linear and the tabular models, as the first
version of this repository did, does not reproduce either arm.

For every configuration the number of latent variables is selected exhaustively
by grouped 3-fold SPXY cross-validation, capped at 30 and at what each fold can
support.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from benchmark_common import (
    SearchConfig, load_dataset, run_over_datasets, score_of, two_phase_search,
    write_outputs,
)

MAX_COMPONENTS = 30


class PLSDA(BaseEstimator, ClassifierMixin):
    """PLS-DA: PLS regression on one-hot targets, decided by the largest score."""

    def __init__(self, n_components: int = 2):
        self.n_components = int(n_components)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        one_hot = np.zeros((len(y), len(self.classes_)), dtype=float)
        for j, cls in enumerate(self.classes_):
            one_hot[np.asarray(y) == cls, j] = 1.0
        n_comp = max(1, min(self.n_components, X.shape[0] - 1, X.shape[1]))
        self.pls_ = PLSRegression(n_components=n_comp, scale=False).fit(X, one_hot)
        return self

    def predict(self, X):
        scores = self.pls_.predict(np.asarray(X, dtype=float))
        return self.classes_[np.argmax(np.atleast_2d(scores), axis=1)]


def make_model(task: str, n_components: int):
    if task == "regression":
        return PLSRegression(n_components=int(n_components), scale=False)
    return PLSDA(n_components=int(n_components))


def safe_upper_bound(X: np.ndarray, folds) -> int:
    """Largest number of components every fold can actually fit."""
    bounds = [min(MAX_COMPONENTS, len(train) - 1, X.shape[1]) for train, _ in folds]
    return max(1, min(bounds))


def make_evaluator(task: str, X: np.ndarray, y: np.ndarray):
    def evaluate(cfg: SearchConfig, folds: List[Tuple[np.ndarray, np.ndarray]]):
        upper = safe_upper_bound(X, folds)
        best_score, best_n = float("inf"), 1
        for n_components in range(1, upper + 1):
            scores = []
            for train_idx, valid_idx in folds:
                pipe = Pipeline(cfg.steps() + [("model", make_model(task, n_components))])
                pipe.fit(X[train_idx], y[train_idx])
                pred = np.asarray(pipe.predict(X[valid_idx])).ravel()
                scores.append(score_of(task, y[valid_idx], pred))
            mean_score = float(np.mean(scores))
            if mean_score < best_score:
                best_score, best_n = mean_score, n_components
        return best_score, {"n_components": best_n}

    return evaluate


def run_one(folder: Path, output_dir: Path, *, task: str, verbose: int) -> Dict[str, Any]:
    X, y, X_test, y_test = load_dataset(folder)

    encoder = None
    if task == "classification":
        encoder = LabelEncoder().fit(np.concatenate([y, y_test]) if y_test is not None else y)
        y = encoder.transform(y)

    best_config, best_params, trace = two_phase_search(
        X, y, make_evaluator(task, X, y), linear=True, verbose=verbose)

    final = Pipeline(best_config.steps() + [
        ("model", make_model(task, best_params["n_components"]))])
    final.fit(X, y)
    y_pred = np.asarray(final.predict(X_test)).ravel()
    if encoder is not None:
        y_pred = encoder.inverse_transform(np.asarray(y_pred, dtype=int))

    return write_outputs(output_dir, folder.name, best_config, best_params,
                         trace, y_pred, y_test, task)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="regression", choices=["regression", "classification"])
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()

    name = "PLS-pp-hpo" if args.task == "regression" else "PLS-DA-pp-hpo"
    out = args.output_dir or f"./results_pls_{args.task}"
    run_over_datasets(
        lambda folder, outdir: run_one(folder, outdir, task=args.task, verbose=args.verbose),
        name, Path(args.data_root), Path(out),
        datasets=args.datasets, limit=args.limit, verbose=args.verbose)


if __name__ == "__main__":
    main()
