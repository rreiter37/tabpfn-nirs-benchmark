# -*- coding: utf-8 -*-
"""
run_xgboost.py — XGBoost arm of the benchmark (XGBoost-pp-hpo).

XGBoost is the one tabular model the manuscript tunes, because its library
defaults (learning rate 0.3, 100 trees) are a genuinely poor starting point on
spectra: the Supplementary Material measures a median test gain of 10.3 % from
the search, against less than two points for TabPFN and CatBoost.

The search is nested: for each of the 30 preprocessing configurations, 30 Optuna
TPE trials are drawn over the eight hyperparameters below, each scored by the
same grouped 3-fold SPXY cross-validation. That is 30 x 3 x 30 = 2700 model fits
per dataset, about 26 minutes of GPU time on the hardware of the paper. The
default here is the published budget; lower --n-trials if you only want a
smoke test, and say so if you report the numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from benchmark_common import (
    SEED, SearchConfig, load_dataset, run_over_datasets, score_of,
    two_phase_search, write_outputs,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Search space of the manuscript: eight parameters, six of them continuous.
SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "n_estimators":     {"type": "int",   "low": 100,  "high": 600, "step": 50},
    "learning_rate":    {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "max_depth":        {"type": "int",   "low": 2,    "high": 10},
    "min_child_weight": {"type": "float", "low": 0.1,  "high": 20.0, "log": True},
    "subsample":        {"type": "float", "low": 0.5,  "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.1,  "high": 1.0},
    "reg_lambda":       {"type": "float", "low": 1e-2, "high": 100.0, "log": True},
    "reg_alpha":        {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
}


def suggest(trial: optuna.trial.Trial) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, spec in SEARCH_SPACE.items():
        if spec["type"] == "int":
            out[name] = trial.suggest_int(name, spec["low"], spec["high"],
                                          step=spec.get("step", 1))
        else:
            out[name] = trial.suggest_float(name, spec["low"], spec["high"],
                                            log=spec.get("log", False))
    return out


def make_model(task: str, params: Dict[str, Any], device: str, n_classes: int = 0):
    common = dict(random_state=SEED, n_jobs=-1, tree_method="hist",
                  device=("cuda" if device == "cuda" else "cpu"), **params)
    if task == "regression":
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:squarederror", **common)
    from xgboost import XGBClassifier
    objective = "binary:logistic" if n_classes <= 2 else "multi:softprob"
    return XGBClassifier(objective=objective, **common)


def make_evaluator(task: str, X: np.ndarray, y: np.ndarray, device: str,
                   n_trials: int, n_classes: int):
    """Nested evaluator: an Optuna study inside every preprocessing config."""

    def evaluate(cfg: SearchConfig, folds: List[Tuple[np.ndarray, np.ndarray]]):
        # The preprocessing is fitted once per fold and reused by every trial:
        # only the model changes, so re-running the transforms 30 times would be
        # wasted work.
        cached = []
        for train_idx, valid_idx in folds:
            steps = cfg.steps()
            transformer = Pipeline(steps) if steps else None
            X_tr = transformer.fit_transform(X[train_idx]) if transformer else X[train_idx]
            X_va = transformer.transform(X[valid_idx]) if transformer else X[valid_idx]
            cached.append((np.asarray(X_tr), y[train_idx], np.asarray(X_va), y[valid_idx]))

        def objective(trial: optuna.trial.Trial) -> float:
            params = suggest(trial)
            scores = []
            for X_tr, y_tr, X_va, y_va in cached:
                model = make_model(task, params, device, n_classes)
                model.fit(X_tr, y_tr)
                scores.append(score_of(task, y_va, model.predict(X_va)))
            return float(np.mean(scores))

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=int(n_trials), catch=(Exception,))
        return float(study.best_value), dict(study.best_params)

    return evaluate


def run_one(folder: Path, output_dir: Path, *, task: str, device: str,
            n_trials: int, verbose: int) -> Dict[str, Any]:
    X, y, X_test, y_test = load_dataset(folder)

    encoder, n_classes = None, 0
    if task == "classification":
        encoder = LabelEncoder().fit(np.concatenate([y, y_test]) if y_test is not None else y)
        y = encoder.transform(y)
        n_classes = int(len(encoder.classes_))

    best_config, best_params, trace = two_phase_search(
        X, y, make_evaluator(task, X, y, device, n_trials, n_classes),
        linear=False, verbose=verbose)

    final = Pipeline(best_config.steps() + [
        ("model", make_model(task, best_params, device, n_classes))])
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
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--n-trials", type=int, default=30,
                   help="Optuna trials per preprocessing configuration (30 in the paper).")
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()

    out = args.output_dir or f"./results_xgboost_{args.task}"
    run_over_datasets(
        lambda folder, outdir: run_one(folder, outdir, task=args.task,
                                       device=args.device, n_trials=args.n_trials,
                                       verbose=args.verbose),
        "XGBoost-pp-hpo", Path(args.data_root), Path(out),
        datasets=args.datasets, limit=args.limit, verbose=args.verbose)


if __name__ == "__main__":
    main()
