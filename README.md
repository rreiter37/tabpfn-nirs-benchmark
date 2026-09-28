# TabPFN NIR benchmark — reproduction scripts

Scripts reproducing the benchmark of *Using tabular foundation models for robust calibration of near-infrared sensing data*, on 66 near-infrared
datasets (54 regression, 12 classification). One script per model arm; they share a
single protocol module and use [nirs4all](https://github.com/GBeurier/nirs4all) for
every signal transform and for the SPXY splitter.

## Layout

| File | Arm of the benchmark |
|---|---|
| `benchmark_common.py` | Protocol: preprocessing spaces, two-phase search, CV, metrics, dataset driver |
| `run_pls.py` | `PLS-pp-hpo` (regression) and `PLS-DA-pp-hpo` (classification), the references |
| `run_ridge.py` | `Ridge-pp-hpo`, regression only |
| `run_catboost.py` | `CatBoost-pp`, fixed hyperparameters |
| `run_xgboost.py` | `XGBoost-pp-hpo`, nested Optuna search |
| `run_tabpfn.py` | `TabPFN-Raw` (`--variant raw`) and `TabPFN-pp` (`--variant pp`) |

## Protocol

For every dataset and every model except TabPFN-Raw, the preprocessing is selected by
a **two-phase search** scored by grouped 3-fold SPXY cross-validation on the
calibration set:

| | phase 1 (shape × scatter) | phase 2 (top 3 extended) | total |
|---|---|---|---|
| tabular models (TabPFN, CatBoost, XGBoost) | 7 × 3 = 21 | × {none, PCA 0.25, OSC} = 9 | **30** |
| linear models (PLS, Ridge, PLS-DA) | 8 × 3 = 24 | × representation (4) × scaler (3) = 36 | **60** |

The linear space is the wider one: it also searches a Gaussian first derivative, and
its phase 2 is a representation × scaler grid. The two spaces are deliberately
different and must not be merged.

The best configuration is then refitted on the whole calibration set and applied to
the external test set. TabPFN-Raw skips the search entirely and is fitted on the raw
spectra, which is what isolates the contribution of preprocessing.

Model hyperparameters, and in particular the values that **differ between the search
and the final refit**:

| model | during cross-validation | final refit |
|---|---|---|
| TabPFN | `n_estimators = 1` | `n_estimators = 16` |
| CatBoost | `iterations = 200` | `iterations = 500` |
| XGBoost | 30 Optuna TPE trials per preprocessing configuration (2700 fits/dataset) | best trial |
| PLS / PLS-DA | exhaustive CV over `n_components`, 1..min(30, n−1, p) | best value |
| Ridge | 30 Optuna TPE trials on `alpha`, log-uniform in [1e-3, 100] | best value |

Everything is seeded at 42.

## Requirements

```bash
pip install -r requirements.txt
```

TabPFN downloads its default checkpoint on first use. The published runs used
`tabpfn-v2.5-regressor-v2.5_real.ckpt`; pass it with `--model-path` to reproduce the
reported numbers exactly, otherwise expect small differences.

## Usage

Each dataset folder holds `Xtrain.csv`, `Ytrain.csv`, `Xtest.csv` and optionally
`Ytest.csv`, semicolon-separated. Point `--data-root` at a folder of such folders;
they are discovered recursively.

```bash
DATA=/path/to/Data/regression

python run_pls.py      --task regression --data-root $DATA
python run_ridge.py                      --data-root $DATA
python run_catboost.py --task regression --data-root $DATA --device GPU
python run_xgboost.py  --task regression --data-root $DATA --device cuda
python run_tabpfn.py   --variant raw --task regression --data-root $DATA --device cuda
python run_tabpfn.py   --variant pp  --task regression --data-root $DATA --device cuda
```

Classification uses the same scripts with `--task classification` (Ridge excepted,
which is not part of that benchmark; `run_pls.py --task classification` runs PLS-DA).

Useful flags: `--datasets NAME [NAME ...]` to restrict the run, `--limit N` to stop
after N datasets, `--output-dir` to choose where results go.

Each run writes, per dataset, the selected configuration (`__best_config.json`), the
full search trace (`__search_results.csv`), the test predictions
(`__final_predictions.csv`), and a `summary_runs.csv` for the whole campaign.

## Notes

- **Cost.** XGBoost is by far the most expensive arm, around 26 min of GPU time per
  dataset at the published budget, because each of its 2700 configurations trains a
  model from scratch. Lower `--n-trials` for a quick check, and say so if you report
  the numbers.
- **Scope.** These scripts reproduce the arms of the main manuscript. The additional
  hyperparameter study of the Supplementary Material, which also tunes TabPFN and
  CatBoost, is not included.
