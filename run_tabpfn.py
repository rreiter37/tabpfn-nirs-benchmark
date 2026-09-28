# -*- coding: utf-8 -*-
"""
run_tabpfn.py — TabPFN arms of the benchmark.

Two variants, both reported in the manuscript:

    --variant raw   TabPFN-Raw: no preprocessing and no search. The calibration
                    spectra go straight into the model, which is why this arm
                    isolates what preprocessing actually buys.
    --variant pp    TabPFN-pp: two-phase preprocessing search (21 + 9 = 30
                    configurations) selected by grouped 3-fold SPXY CV.

TabPFN is training-free, so the only hyperparameter that moves is the size of
the inference ensemble, and it differs between the two stages:

    n_estimators = 1   while scoring preprocessing configurations
    n_estimators = 16  for the final refit and the reported predictions

Scoring the search with 16 estimators, as the first version of this repository
did, multiplies the search cost by 16 for a ranking of preprocessing pipelines
that barely changes.
"""

from __future__ import annotations

# Keep torch.compile out of the way on machines without a working compiler.
try:  # pragma: no cover
    import torch  # noqa: F401
    import torch._dynamo  # noqa: F401
except Exception:
    pass

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

N_ESTIMATORS_SEARCH = 1
N_ESTIMATORS_FINAL = 16


def make_model(task: str, n_estimators: int, device: str, model_path: Optional[str]):
    """Build a TabPFN estimator for the requested task."""
    kwargs: Dict[str, Any] = dict(n_estimators=int(n_estimators), device=device,
                                  random_state=SEED, ignore_pretraining_limits=True)
    if model_path:
        kwargs["model_path"] = str(model_path)
    if task == "regression":
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(**kwargs)
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(**kwargs)


def make_evaluator(task: str, X: np.ndarray, y: np.ndarray, device: str,
                   model_path: Optional[str]):
    """Return the CV evaluator used by the two-phase search."""

    def evaluate(cfg: SearchConfig, folds: List[Tuple[np.ndarray, np.ndarray]]):
        scores = []
        for train_idx, valid_idx in folds:
            pipe = Pipeline(cfg.steps() + [
                ("model", make_model(task, N_ESTIMATORS_SEARCH, device, model_path))])
            pipe.fit(X[train_idx], y[train_idx])
            scores.append(score_of(task, y[valid_idx], pipe.predict(X[valid_idx])))
        return float(np.mean(scores)), {"n_estimators": N_ESTIMATORS_SEARCH}

    return evaluate


def run_one(folder: Path, output_dir: Path, *, variant: str, task: str,
            device: str, model_path: Optional[str], verbose: int) -> Dict[str, Any]:
    X, y, X_test, y_test = load_dataset(folder)

    encoder = None
    if task == "classification":
        encoder = LabelEncoder().fit(np.concatenate([y, y_test]) if y_test is not None else y)
        y = encoder.transform(y)

    if variant == "raw":
        best_config, best_params, trace = SearchConfig(), {}, []
    else:
        best_config, best_params, trace = two_phase_search(
            X, y, make_evaluator(task, X, y, device, model_path),
            linear=False, verbose=verbose)

    final = Pipeline(best_config.steps() + [
        ("model", make_model(task, N_ESTIMATORS_FINAL, device, model_path))])
    final.fit(X, y)
    y_pred = final.predict(X_test)

    y_true = y_test
    if encoder is not None:
        y_pred = encoder.inverse_transform(np.asarray(y_pred, dtype=int))

    return write_outputs(output_dir, folder.name, best_config,
                         {**best_params, "n_estimators_final": N_ESTIMATORS_FINAL},
                         trace, y_pred, y_true, task)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=["raw", "pp"])
    p.add_argument("--task", default="regression", choices=["regression", "classification"])
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--model-path", default=None,
                   help="Optional TabPFN checkpoint. The published runs used "
                        "tabpfn-v2.5-regressor-v2.5_real.ckpt; without it the "
                        "library default is used and numbers may differ slightly.")
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()

    name = f"TabPFN-{'Raw' if args.variant == 'raw' else 'pp'}"
    out = args.output_dir or f"./results_tabpfn_{args.variant}_{args.task}"
    run_over_datasets(
        lambda folder, outdir: run_one(folder, outdir, variant=args.variant,
                                       task=args.task, device=args.device,
                                       model_path=args.model_path, verbose=args.verbose),
        name, Path(args.data_root), Path(out),
        datasets=args.datasets, limit=args.limit, verbose=args.verbose)


if __name__ == "__main__":
    main()
