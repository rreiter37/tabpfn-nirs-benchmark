#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_common.py

Shared building blocks for the *nirs4all-idiomatic* reproduction of the TabPFN NIR
benchmark. The original study used custom orchestration scripts; here every
experiment is expressed with the native nirs4all pipeline formalism (a pipeline is
a list of steps) and executed with ``nirs4all.run``.

The preprocessing search is expressed the idiomatic nirs4all way: each
preprocessing axis is an ``{"_or_": [...]}`` step, so nirs4all expands the pipeline
into the Cartesian product of preprocessing chains, evaluates each by SPXY-grouped
cross-validation on the calibration set, and selects the best configuration before
refitting and predicting on the external test set.

Datasets follow the benchmark layout (``Xtrain.csv``, ``Ytrain.csv``, ``Xtest.csv``,
``Ytest.csv`` per folder), which nirs4all loads directly from a folder path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.operators.transforms import (
    ASLSBaseline,
    AreaNormalization,
    ExtendedMultiplicativeScatterCorrection,
    FlexiblePCA,
    Gaussian,
    Haar,
    IdentityTransformer,
    OSC,
    SavitzkyGolay,
    StandardNormalVariate,
)
from nirs4all.operators.splitters import SPXYGFold

SEED = 42
N_SPLITS = 3


# --------------------------------------------------------------------------- #
# Preprocessing search space (matches the benchmark's operator families)
# --------------------------------------------------------------------------- #
def _sg(name: str) -> SavitzkyGolay:
    """Build a Savitzky-Golay transform from a ``SG_<window>_<polyorder>_<deriv>`` tag."""
    _, window, poly, deriv = name.split("_")
    return SavitzkyGolay(window_length=int(window), polyorder=int(poly), deriv=int(deriv))


# A bare IdentityTransformer() encodes the "None" (no-op) option of each axis.
SHAPE_OPS: List[Any] = [
    IdentityTransformer(),
    ASLSBaseline(),
    _sg("SG_11_2_1"), _sg("SG_15_2_1"), _sg("SG_21_2_1"),
    _sg("SG_15_3_2"), _sg("SG_21_3_2"),
    Gaussian(order=1, sigma=2),
]
SCATTER_OPS: List[Any] = [
    IdentityTransformer(),
    StandardNormalVariate(),
    ExtendedMultiplicativeScatterCorrection(),
]
# Final-representation axis used by the linear models (PLS/Ridge).
REPR_OPS: List[Any] = [
    IdentityTransformer(),
    Haar(),
    AreaNormalization(),
    OSC(),
]
SCALER_OPS: List[Any] = [
    IdentityTransformer(),
    StandardScaler(),
    MinMaxScaler(),
]
# Phase-2 axis used by the nonlinear models (TabPFN/CatBoost/CNN): PCA keeping a
# fraction of the components, or OSC.
PHASE2_OPS: List[Any] = [
    IdentityTransformer(),
    FlexiblePCA(n_components=0.25),
    OSC(),
]


def or_step(options: List[Any]) -> Dict[str, Any]:
    """Wrap a list of operator options as an nirs4all ``_or_`` search step."""
    return {"_or_": list(options)}


def linear_preprocessing_search() -> List[Dict[str, Any]]:
    """Preprocessing search steps for PLS/Ridge (shape x scatter x repr x scaler)."""
    return [or_step(SHAPE_OPS), or_step(SCATTER_OPS), or_step(REPR_OPS), or_step(SCALER_OPS)]


def nonlinear_preprocessing_search() -> List[Dict[str, Any]]:
    """Preprocessing search steps for TabPFN/CatBoost (shape x scatter x phase2)."""
    return [or_step(SHAPE_OPS), or_step(SCATTER_OPS), or_step(PHASE2_OPS)]


def conv_preprocessing_search() -> List[Dict[str, Any]]:
    """Preprocessing search steps for the 1D-CNN (shape x scatter only).

    The phase-2 axis (PCA / OSC) is intentionally excluded: PCA discards the
    ordered spectral axis that a 1D convolution is designed to exploit, so it is
    not a meaningful preprocessing for a convolutional model.
    """
    return [or_step(SHAPE_OPS), or_step(SCATTER_OPS)]


def cv_splitter() -> SPXYGFold:
    """SPXY-grouped K-fold used for calibration-set cross-validation."""
    return SPXYGFold(n_splits=N_SPLITS, random_state=SEED)


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
def is_dataset_dir(path: Path) -> bool:
    """A dataset folder holds at least Xtrain/Ytrain/Xtest files."""
    names = {p.name for p in path.iterdir()} if path.is_dir() else set()
    return {"Xtrain.csv", "Ytrain.csv", "Xtest.csv"}.issubset(names)


def find_datasets(data_root: Path) -> List[Path]:
    """Recursively list benchmark dataset folders under ``data_root``."""
    return sorted(p for p in data_root.rglob("*") if p.is_dir() and is_dataset_dir(p))


# --------------------------------------------------------------------------- #
# Driver: run one model pipeline over every dataset
# --------------------------------------------------------------------------- #
def run_over_datasets(
    build_pipeline,
    model_name: str,
    data_root: Path,
    output_dir: Path,
    datasets: List[str] | None = None,
    limit: int | None = None,
    verbose: int = 1,
) -> None:
    """Run ``build_pipeline()`` on every dataset and write a summary CSV.

    Full per-configuration results, predictions and the selected pipeline are
    saved by nirs4all itself (``save_artifacts=True``); this driver only iterates
    datasets and collects the best test score per dataset.
    """
    import time
    import nirs4all
    import pandas as pd

    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if datasets:
        ds = [data_root / d if not Path(d).is_absolute() else Path(d) for d in datasets]
    else:
        ds = find_datasets(data_root)
    if limit:
        ds = ds[:limit]

    rows: List[Dict[str, Any]] = []
    for i, d in enumerate(ds, 1):
        t0 = time.time()
        try:
            res = nirs4all.run(
                pipeline=build_pipeline(),
                dataset=str(d),
                name=f"{model_name}__{d.name}",
                verbose=verbose,
                save_artifacts=True,
                save_charts=False,
                random_state=SEED,
            )
            rows.append({
                "dataset": d.name, "model": model_name, "status": "ok",
                "rmsep": getattr(res, "best_rmse", None),
                "r2_test": getattr(res, "best_r2", None),
                "accuracy": getattr(res, "best_accuracy", None),
                "elapsed_sec": round(time.time() - t0, 2),
            })
            print(f"[{i}/{len(ds)}] {model_name} {d.name}: "
                  f"RMSEP={getattr(res, 'best_rmse', None)} ({time.time()-t0:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            rows.append({"dataset": d.name, "model": model_name, "status": "error",
                         "error": str(exc)[:300], "elapsed_sec": round(time.time() - t0, 2)})

    out_csv = output_dir / f"{model_name}_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}", flush=True)
