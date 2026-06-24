#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_tabpfn.py — TabPFN calibration, nirs4all-idiomatic.

TabPFN is training-free: the pretrained model is passed as a scikit-learn-compatible
estimator inside an nirs4all pipeline.

  --variant raw : direct in-context inference on raw spectra (no preprocessing,
                  no search) -- fit on the calibration set, predict the test set.
  --variant opt : the same pretrained model wrapped in the preprocessing search
                  (shape x scatter x phase-2), selected by SPXY-grouped CV.

Example
-------
    python run_tabpfn.py --variant opt --data-root ../../Data/regression \\
        --model-path ../../tabpfn-v2.5-regressor-v2.5_real.ckpt --device cuda
"""

from __future__ import annotations

import os

# nirs4all (imported via benchmark_common) pulls in TensorFlow; torch's lazy
# ``import torch._dynamo`` (triggered by TabPFN) then segfaults if TF was imported
# first. Pre-load torch._dynamo here, before TabPFN and before benchmark_common.
# (This is only needed for the torch-based TabPFN scripts.)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
try:  # pragma: no cover - environment hardening
    import torch  # noqa: F401
    import torch._dynamo  # noqa: F401
except Exception:
    pass

import argparse
from pathlib import Path
from typing import Optional

from tabpfn import TabPFNRegressor

from benchmark_common import (
    SEED, cv_splitter, nonlinear_preprocessing_search, run_over_datasets,
)


def make_tabpfn(model_path: Optional[str], device: str, n_estimators: int) -> TabPFNRegressor:
    kwargs = dict(
        n_estimators=int(n_estimators),
        device=device,
        random_state=SEED,
        ignore_pretraining_limits=True,
    )
    if model_path:
        kwargs["model_path"] = str(model_path)
    return TabPFNRegressor(**kwargs)


def build_pipeline(variant: str, model_path: Optional[str], device: str, n_estimators: int) -> list:
    model_step = {"model": make_tabpfn(model_path, device, n_estimators),
                  "name": f"TabPFN-{variant}"}
    if variant == "raw":
        # No preprocessing and no search: fit on calibration, predict on test.
        return [model_step]
    # variant == "opt": preprocessing search selected by SPXY-grouped CV.
    return [*nonlinear_preprocessing_search(), cv_splitter(), model_step]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=["raw", "opt"])
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model-path", default=None,
                   help="Path to the TabPFN checkpoint (default: TabPFN's bundled model).")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--n-estimators", type=int, default=16)
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()
    out = args.output_dir or f"./results_tabpfn_{args.variant}"
    run_over_datasets(
        lambda: build_pipeline(args.variant, args.model_path, args.device, args.n_estimators),
        f"TabPFN-{args.variant}", Path(args.data_root), Path(out),
        datasets=args.datasets, limit=args.limit, verbose=args.verbose)


if __name__ == "__main__":
    main()
