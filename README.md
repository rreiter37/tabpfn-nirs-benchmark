# TabPFN NIR benchmark — nirs4all reproduction

Reproduction of the calibration experiments from the article *"Tabular foundation
models for robust calibration of near-infrared chemical sensing data"*, written
with the native [**nirs4all**](https://github.com/GBeurier/nirs4all) pipeline
formalism. Each model is expressed as an nirs4all pipeline (a list of steps) and
executed with `nirs4all.run`.

## Layout

| File | Model |
|---|---|
| `benchmark_common.py` | Shared preprocessing search space, SPXY splitter, dataset discovery, run driver |
| `run_pls.py` | PLS (latent variables tuned by CV) |
| `run_ridge.py` | Ridge (`alpha` tuned by CV, log scale) |
| `run_catboost.py` | CatBoost (fixed trees, GPU) |
| `run_cnn.py` | 1D-CNN / native NICON (architecture search, GPU) |
| `run_tabpfn.py` | TabPFN, `--variant raw` (no preprocessing) or `--variant opt` (preprocessing search) |

## Preprocessing search space

Expressed as `{"_or_": [...]}` steps, one per axis, so nirs4all evaluates every
combination by SPXY-grouped 3-fold CV and selects the best:

- **shape**: none, ASLS baseline, Savitzky–Golay (5 window/order/derivative settings), Gaussian(1, 2)
- **scatter**: none, SNV, EMSC
- **representation** (PLS/Ridge): none, Haar, area normalization, OSC
- **scaler** (PLS/Ridge): none, StandardScaler, MinMaxScaler
- **phase-2** (TabPFN/CatBoost): none, PCA (25% of components), OSC

`IdentityTransformer()` encodes the "none" option of each axis. The CNN-1D searches
over **shape × scatter only**: the phase-2 axis is excluded because PCA discards the
ordered spectral axis that a 1D convolution is designed to exploit.

## Requirements

- `nirs4all` (>= 0.8), `scikit-learn`, `catboost`, `tensorflow` (for NICON),
  `tabpfn` and `torch` (for TabPFN).
- The NIR datasets, in the per-folder layout `Xtrain.csv`, `Ytrain.csv`,
  `Xtest.csv`, `Ytest.csv` (the public benchmark datasets; see the article's
  Supplementary Material for the source and access link of each dataset).

## Usage

```bash
# Regression datasets root: a folder of dataset subfolders
DATA=../../Data/regression

python run_pls.py      --data-root $DATA --output-dir ./results_pls
python run_ridge.py    --data-root $DATA --output-dir ./results_ridge
python run_catboost.py --data-root $DATA --output-dir ./results_catboost --device GPU
python run_cnn.py      --data-root $DATA --output-dir ./results_cnn

# TabPFN (training-free); point --model-path to the TabPFN-2.5 checkpoint
python run_tabpfn.py --variant raw --data-root $DATA --device cuda \
    --model-path /path/to/tabpfn-v2.5-regressor-v2.5_real.ckpt
python run_tabpfn.py --variant opt --data-root $DATA --device cuda \
    --model-path /path/to/tabpfn-v2.5-regressor-v2.5_real.ckpt
```

Useful flags (all scripts): `--datasets <name> [<name> ...]` to restrict to specific
dataset folders, `--limit N` to process only the first N datasets, `--verbose 0/1`.

Each script writes a `<Model>_summary.csv` (best test RMSEP / R² per dataset) to
`--output-dir`; nirs4all additionally saves the full per-configuration results,
selected pipelines and predictions to its workspace (`save_artifacts=True`).

## Notes

- Cross-validation uses `SPXYGFold` (SPXY-grouped K-fold) with `random_state=42`,
  matching the study; the fixed calibration/test split is the one shipped with each
  dataset (Kennard–Stone or SPXY, as named in the dataset folder).
- The full Cartesian preprocessing search is intentionally exhaustive. To reduce
  runtime, trim the operator lists in `benchmark_common.py`.
- TabPFN and CatBoost use the GPU; PLS and Ridge run on CPU. The native NICON is a
  TensorFlow model: on GPUs whose TensorFlow build lacks the required kernels
  (e.g. `CUDA_ERROR_INVALID_HANDLE` on very recent architectures), run the CNN on
  CPU with `python run_cnn.py --cpu ...` (or set `N4A_CNN_CPU=1`).
- Each `nirs4all.run` creates a local `workspace/` (DuckDB store) in the current
  working directory. Run the scripts **one at a time**, or from separate working
  directories, so concurrent runs do not share the same `workspace/`.
