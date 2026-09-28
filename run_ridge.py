# -*- coding: utf-8 -*-
"""
run_ridge.py — Ridge arm of the benchmark (Ridge-pp-hpo), regression only.

Ridge shares the wide linear preprocessing space with PLS (24 + 3 x 12 = 60
configurations, Gaussian derivative included), and its single hyperparameter is
selected by 30 Optuna TPE trials over a log-uniform alpha in [1e-3, 100], nested
inside every configuration and scored by the same grouped 3-fold SPXY CV.

Ridge is not part of the classification benchmark, where PLS-DA is the linear
reference; see run_pls.py --task classification.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from benchmark_common import (
    SEED, SearchConfig, load_dataset, run_over_datasets, score_of,
    two_phase_search, write_outputs,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

ALPHA_MIN, ALPHA_MAX = 1e-3, 100.0


def make_evaluator(X: np.ndarray, y: np.ndarray, n_trials: int):
    def evaluate(cfg: SearchConfig, folds: List[Tuple[np.ndarray, np.ndarray]]):
        # Preprocessing fitted once per fold and reused across trials: only alpha
        # changes, so re-transforming the spectra 30 times would be wasted work.
        cached = []
        for train_idx, valid_idx in folds:
            steps = cfg.steps()
            transformer = Pipeline(steps) if steps else None
            X_tr = transformer.fit_transform(X[train_idx]) if transformer else X[train_idx]
            X_va = transformer.transform(X[valid_idx]) if transformer else X[valid_idx]
            cached.append((np.asarray(X_tr), y[train_idx], np.asarray(X_va), y[valid_idx]))

        def objective(trial: optuna.trial.Trial) -> float:
            alpha = trial.suggest_float("alpha", ALPHA_MIN, ALPHA_MAX, log=True)
            scores = []
            for X_tr, y_tr, X_va, y_va in cached:
                model = Ridge(alpha=alpha, random_state=SEED).fit(X_tr, y_tr)
                scores.append(score_of("regression", y_va, model.predict(X_va)))
            return float(np.mean(scores))

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=int(n_trials), catch=(Exception,))
        return float(study.best_value), dict(study.best_params)

    return evaluate


def run_one(folder: Path, output_dir: Path, *, n_trials: int, verbose: int) -> Dict[str, Any]:
    X, y, X_test, y_test = load_dataset(folder)

    best_config, best_params, trace = two_phase_search(
        X, y, make_evaluator(X, y, n_trials), linear=True, verbose=verbose)

    final = Pipeline(best_config.steps() + [
        ("model", Ridge(alpha=float(best_params["alpha"]), random_state=SEED))])
    final.fit(X, y)
    y_pred = np.asarray(final.predict(X_test)).ravel()

    return write_outputs(output_dir, folder.name, best_config, best_params,
                         trace, y_pred, y_test, "regression")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="./results_ridge_regression")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-trials", type=int, default=30,
                   help="Optuna trials per preprocessing configuration (30 in the paper).")
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()

    run_over_datasets(
        lambda folder, outdir: run_one(folder, outdir, n_trials=args.n_trials,
                                       verbose=args.verbose),
        "Ridge-pp-hpo", Path(args.data_root), Path(args.output_dir),
        datasets=args.datasets, limit=args.limit, verbose=args.verbose)


if __name__ == "__main__":
    main()
