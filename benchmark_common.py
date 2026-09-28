# -*- coding: utf-8 -*-
"""
benchmark_common.py — shared protocol for the TabPFN-NIRS benchmark.

Everything the models have in common lives here: the preprocessing search
spaces, the two-phase search itself, the cross-validation splitter, the metrics
and the dataset driver. Each ``run_<model>.py`` only supplies the model and how
to fit it.

Protocol reproduced from the manuscript
---------------------------------------
For every dataset, and for every model except TabPFN-Raw:

  Phase 1   shape x scatter, evaluated by grouped 3-fold SPXY cross-validation
            on the calibration set.
              - tabular models (TabPFN, CatBoost, XGBoost): 7 shapes x 3 = 21
              - linear models (PLS, Ridge, PLS-DA):         8 shapes x 3 = 24
            The linear space is the one that also contains the Gaussian
            derivative; the tabular space does not.

  Phase 2   the best ``PHASE1_TOP_K`` = 3 configurations are extended:
              - tabular: x {None, PCA(0.25), OSC}            ->  9 more, 30 total
              - linear:  x repr(4) x scaler(3)               -> 36 more, 60 total

  Final     the single best configuration is refitted on the whole calibration
            set and applied to the external test set.

TabPFN-Raw skips the search entirely: it is fitted on the raw calibration
spectra and applied to the raw test spectra.

Model hyperparameters follow the manuscript description (cf. table 1 and the Supplementary Material):

    TabPFN     n_estimators   1 during CV, 16 for the final refit
    CatBoost   iterations   200 during CV, 500 for the final refit
    XGBoost    30 Optuna TPE trials per preprocessing configuration
    PLS/PLS-DA exhaustive CV over n_components (1..min(30, n-1, p))
    Ridge      30 Optuna TPE trials on alpha, log-uniform in [1e-3, 100]
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.operators.splitters import SPXYGFold
from nirs4all.operators.transforms import (
    ASLSBaseline,
    AreaNormalization,
    Gaussian,
    Haar,
    IdentityTransformer,
    OSC,
    SavitzkyGolay,
    StandardNormalVariate,
)

# --------------------------------------------------------------------------- #
# Protocol constants
# --------------------------------------------------------------------------- #
SEED = 42
N_SPLITS = 3
PHASE1_TOP_K = 3

# Shapes available to the tabular models. Gaussian is absent on purpose.
SHAPES_TABULAR = ["None", "ASLSBaseline", "SG_11_2_1", "SG_15_2_1",
                  "SG_21_2_1", "SG_15_3_2", "SG_21_3_2"]
# Linear models additionally search a Gaussian first derivative.
SHAPES_LINEAR = SHAPES_TABULAR + ["Gaussian_1_2"]

SCATTERS = ["None", "SNV", "EMSC"]
PHASE2_TABULAR = ["None", "PCA_features_0.25", "OSC"]
REPRS_LINEAR = ["None", "Haar", "AreaNormalization", "OSC"]
SCALERS_LINEAR = ["None", "StandardScaler", "MinMaxScaler"]


# --------------------------------------------------------------------------- #
# Corrected EMSC
# --------------------------------------------------------------------------- #
class EMSCadapted(TransformerMixin, BaseEstimator):
    """Extended multiplicative scatter correction with an uncentred reference.
    The published results use this version of the preprocessing
    """

    def __init__(self, degree: int = 2):
        self.degree = int(degree)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.reference_ = X.mean(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def _design(self) -> np.ndarray:
        n = self.n_features_in_
        axis = np.linspace(-1.0, 1.0, n)
        columns = [np.ones(n), self.reference_]
        columns += [axis ** k for k in range(1, self.degree + 1)]
        return np.vstack(columns).T

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        design = self._design()
        coefficients, *_ = np.linalg.lstsq(design, X.T, rcond=None)
        baseline = design[:, [0]] @ coefficients[[0], :]
        polynomial = design[:, 2:] @ coefficients[2:, :]
        scatter = coefficients[1, :]
        scatter = np.where(np.abs(scatter) < 1e-12, 1.0, scatter)
        return ((X.T - baseline - polynomial) / scatter).T


# --------------------------------------------------------------------------- #
# Preprocessing steps
# --------------------------------------------------------------------------- #
def _savitzky_golay(tag: str) -> SavitzkyGolay:
    _, window, poly, deriv = tag.split("_")
    return SavitzkyGolay(window_length=int(window), polyorder=int(poly),
                         deriv=int(deriv))


def _make_operator(name: str) -> Optional[Any]:
    """Instantiate one transform from its protocol name; ``None`` is a no-op."""
    if name in (None, "None"):
        return None
    if name.startswith("SG_"):
        return _savitzky_golay(name)
    return {
        "ASLSBaseline": lambda: ASLSBaseline(),
        "Gaussian_1_2": lambda: Gaussian(order=1, sigma=2),
        "SNV": lambda: StandardNormalVariate(),
        "EMSC": lambda: EMSCadapted(),
        "PCA_features_0.25": lambda: PCA(n_components=0.25, random_state=SEED),
        "OSC": lambda: OSC(n_components=1),
        "Haar": lambda: Haar(),
        "AreaNormalization": lambda: AreaNormalization(),
        "StandardScaler": lambda: StandardScaler(),
        "MinMaxScaler": lambda: MinMaxScaler(),
        "Identity": lambda: IdentityTransformer(),
    }[name]()


@dataclass(frozen=True)
class SearchConfig:
    """One preprocessing pipeline, named by its operators."""

    shape: str = "None"
    scatter: str = "None"
    phase2: str = "None"          # tabular models
    final_repr: str = "None"      # linear models
    scaler: str = "None"          # linear models

    def steps(self) -> List[Tuple[str, Any]]:
        """sklearn ``Pipeline`` steps for this configuration, in order."""
        wanted = [("shape", self.shape), ("scatter", self.scatter),
                  ("phase2", self.phase2), ("repr", self.final_repr),
                  ("scaler", self.scaler)]
        out = []
        for slot, name in wanted:
            operator = _make_operator(name)
            if operator is not None:
                out.append((slot, operator))
        return out


def enumerate_phase1(linear: bool) -> List[SearchConfig]:
    """Phase-1 grid: 24 configurations for linear models, 21 for tabular ones."""
    shapes = SHAPES_LINEAR if linear else SHAPES_TABULAR
    return [SearchConfig(shape=s, scatter=c) for s in shapes for c in SCATTERS]


def enumerate_phase2(bases: Sequence[SearchConfig], linear: bool) -> List[SearchConfig]:
    """Phase-2 extensions of the retained phase-1 configurations."""
    out: List[SearchConfig] = []
    for base in bases:
        if linear:
            for repr_name in REPRS_LINEAR:
                for scaler in SCALERS_LINEAR:
                    out.append(SearchConfig(shape=base.shape, scatter=base.scatter,
                                            final_repr=repr_name, scaler=scaler))
        else:
            for op in PHASE2_TABULAR:
                out.append(SearchConfig(shape=base.shape, scatter=base.scatter,
                                        phase2=op))
    return out


# --------------------------------------------------------------------------- #
# Cross-validation and metrics
# --------------------------------------------------------------------------- #
def cv_folds(X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Grouped 3-fold SPXY split of the calibration set, fixed by ``SEED``."""
    splitter = SPXYGFold(n_splits=N_SPLITS, random_state=SEED)
    return [(tr, va) for tr, va in splitter.split(X, y)]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def score_of(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Score to be *minimised*, whatever the task."""
    if task == "regression":
        return rmse(y_true, y_pred)
    return -balanced_accuracy(y_true, y_pred)


# --------------------------------------------------------------------------- #
# Two-phase preprocessing search
# --------------------------------------------------------------------------- #
Evaluator = Callable[[SearchConfig, List[Tuple[np.ndarray, np.ndarray]]],
                     Tuple[float, Dict[str, Any]]]


def two_phase_search(
    X: np.ndarray,
    y: np.ndarray,
    evaluate: Evaluator,
    linear: bool,
    top_k: int = PHASE1_TOP_K,
    verbose: int = 1,
) -> Tuple[SearchConfig, Dict[str, Any], List[Dict[str, Any]]]:
    """Run the phase-1 / phase-2 search and return the winning configuration.

    ``evaluate`` receives a configuration and the CV folds, and returns the mean
    cross-validated score to minimise together with the model hyperparameters it
    selected for that configuration.
    """
    folds = cv_folds(X, y)
    trace: List[Dict[str, Any]] = []

    def run(configs: Sequence[SearchConfig], phase: int) -> List[Dict[str, Any]]:
        rows = []
        for i, cfg in enumerate(configs, 1):
            try:
                score, params = evaluate(cfg, folds)
            except Exception as exc:                # a configuration may be invalid
                if verbose:
                    print(f"    [phase {phase}] {i}/{len(configs)} {cfg} -> failed: {exc}")
                continue
            rows.append({"phase": phase, "config": cfg, "score": score, "params": params})
            if verbose > 1:
                print(f"    [phase {phase}] {i}/{len(configs)} {cfg} -> {score:.6g}")
        return rows

    phase1 = run(enumerate_phase1(linear), 1)
    if not phase1:
        raise RuntimeError("every phase-1 configuration failed")
    phase1.sort(key=lambda r: r["score"])
    trace += phase1

    bases = [r["config"] for r in phase1[:top_k]]
    phase2 = run(enumerate_phase2(bases, linear), 2)
    trace += phase2

    best = min(trace, key=lambda r: r["score"])
    if verbose:
        print(f"    best: {best['config']} score={best['score']:.6g}")
    return best["config"], best["params"], trace


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def _read(path: Path) -> pd.DataFrame:
    """Read one benchmark CSV (';' separated, '.' decimal)."""
    return pd.read_csv(path, sep=";", decimal=".")


def is_dataset_dir(path: Path) -> bool:
    names = {p.name for p in path.iterdir()} if path.is_dir() else set()
    return {"Xtrain.csv", "Ytrain.csv", "Xtest.csv"}.issubset(names)


def find_datasets(data_root: Path) -> List[Path]:
    return sorted(p for p in data_root.rglob("*") if p.is_dir() and is_dataset_dir(p))


def load_dataset(folder: Path):
    """Return ``(Xtrain, ytrain, Xtest, ytest_or_None)`` as numpy arrays."""
    xtr = _read(folder / "Xtrain.csv").to_numpy(dtype=float)
    ytr = _read(folder / "Ytrain.csv").iloc[:, 0].to_numpy()
    xte = _read(folder / "Xtest.csv").to_numpy(dtype=float)
    yte_path = folder / "Ytest.csv"
    yte = _read(yte_path).iloc[:, 0].to_numpy() if yte_path.exists() else None
    return xtr, ytr, xte, yte


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_over_datasets(
    run_one: Callable[[Path, Path], Dict[str, Any]],
    model_name: str,
    data_root: Path,
    output_dir: Path,
    datasets: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    verbose: int = 1,
) -> None:
    """Apply ``run_one`` to every dataset and write a run summary."""
    folders = find_datasets(Path(data_root))
    if datasets:
        wanted = {str(d) for d in datasets}
        folders = [f for f in folders if f.name in wanted]
    if limit:
        folders = folders[: int(limit)]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    print(f"{model_name}: {len(folders)} dataset(s)", flush=True)
    for i, folder in enumerate(folders, 1):
        started = time.time()
        print(f"[{i}/{len(folders)}] {folder.name}", flush=True)
        try:
            record = run_one(folder, output_dir)
            status = "ok"
        except Exception as exc:
            record, status = {"error": repr(exc)}, "failed"
            print(f"    failed: {exc}", flush=True)
        elapsed = time.time() - started
        summary_rows.append({"model": model_name, "dataset": folder.name,
                             "status": status, "elapsed_sec": round(elapsed, 2),
                             **{k: v for k, v in record.items() if np.isscalar(v)}})
        pd.DataFrame(summary_rows).to_csv(output_dir / "summary_runs.csv", index=False)
        if verbose:
            print(f"    {status} in {elapsed:.1f}s", flush=True)
    print("done.", flush=True)


def write_outputs(
    output_dir: Path,
    dataset: str,
    best_config: SearchConfig,
    best_params: Dict[str, Any],
    trace: List[Dict[str, Any]],
    y_pred: np.ndarray,
    y_true: Optional[np.ndarray],
    task: str,
) -> Dict[str, Any]:
    """Persist the per-dataset artefacts and return the summary fields."""
    out = Path(output_dir)
    (out / f"{dataset}__best_config.json").write_text(
        json.dumps({"config": asdict(best_config), "model_params": best_params},
                   indent=2, default=str), encoding="utf-8")

    pd.DataFrame([{**asdict(r["config"]), "phase": r["phase"], "score": r["score"]}
                  for r in trace]).to_csv(out / f"{dataset}__search_results.csv",
                                          sep=";", index=False)

    preds = pd.DataFrame({"y_pred": y_pred})
    if y_true is not None:
        preds.insert(0, "y_true", y_true)
    preds.to_csv(out / f"{dataset}__final_predictions.csv", sep=";", index=False)

    if y_true is None:
        return {}
    if task == "regression":
        return {"RMSEP": rmse(y_true, y_pred)}
    return {"balanced_accuracy_test": balanced_accuracy(y_true, y_pred)}
