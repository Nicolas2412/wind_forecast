# Wind Forecasting

This repository provides a full training and evaluation pipeline for offshore wind production forecasting.

## What is included

- Unified preprocessing and feature engineering
- Temporal split for robust evaluation
- Portfolio-level and site-level metrics
- Model saving/loading for repeated experiments
- One command entrypoint for all supported model families
- Config-driven backbone for schema, preprocessing, split, and postprocessing

## Supported models

- `random_forest`
- `xgboost`
- `lightgbm`
- `knn`
- `lstm`
- `transformer`

## Clean structure

```text
wind_forecast/
├── main.py
├── tools.py
├── config.yaml
├── requirements.txt
├── README.md
├── optimize.py
├── viz.py
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── deep_models.py
│   │   ├── knn_model.py
│   │   └── tree_models.py
│   └── optimization/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── sequence.py
│       └── tabular.py
└── data/
    ├── dataset_1.parquet
    ├── dataset_2.parquet
    ├── dataset_3.parquet
    └── models/
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Place input parquet files in `data/`.

Expected files:

- `data/dataset_1.parquet`
- `data/dataset_2.parquet`
- `data/dataset_3.parquet`

## Main pipeline

Use `main.py` for both training and evaluation.

```bash
python main.py -m <model_type> -n <run_name>
```

Examples:

```bash
python main.py -m knn -n knn_baseline
python main.py -m lightgbm -n lgbm_v1
python main.py -m lstm -n lstm_v1 --seq_len 48
python main.py -m transformer -n transformer_v1 --no_cv
python main.py -m random_forest -n rf_site0 -u -s 0
python main.py -m xgboost -n xgb_no_prod -dp
python main.py -m knn -n knn_existing --skip_train
```

## Main CLI options

- `-m`, `--model`: model type
- `-n`, `--name`: output model file name
- `-ts`, `--test_size`: test split ratio
- `-u`, `--unique_site`: run on one site only
- `-s`, `--site_index`: selected site index when `-u` is enabled
- `--on_nine_sites`: train on nine sites and test on one held-out site
- `--single_group`: generic alias of single-group mode
- `--holdout_group`: generic alias of holdout-group mode
- `-dp`, `--drop_prod`: remove production-lag features
- `--no_cv`: skip deep-model validation split strategy
- `--skip_train`: load model and evaluate directly
- `--seq_len`: sequence length for LSTM and Transformer
- `-v`, `--verbose`: verbose execution mode
- `--data_folder`: dataset folder override

## Outputs

Models are stored in:

- `data/models/<model_type>/all_sites/<name>.pkl`
- `data/models/<model_type>/site_<idx>/<name>.pkl`

The run prints:

- Global MAE, RMSE, nRMSE
- Site-level MAE, RMSE, nRMSE
- Portfolio total and per-site aggregated metrics

## Hyperparameter optimization

Use `optimize.py` to tune and train tabular or sequence models.

**Tabular optimization:**

```bash
python optimize.py --family tabular --mode all --model all --n_trials 50
```

**Sequence optimization:**

```bash
python optimize.py --family sequence --sequence_model lstm --seq_lengths 6,12,24,48 --drop_prod_options false,true --sequence_site_start 3 --sequence_site_end 9
```

**Both families in one command:**

```bash
python optimize.py --family all --mode all --model all --n_trials 50 --sequence_model lstm --seq_lengths 6,12,24,48 --drop_prod_options false,true --sequence_site_start 3 --sequence_site_end 9
```

## Backbone Configuration

All use-case specific behavior is now centralized in `config.yaml` under the `backbone` section:

- `backbone.schema`: names of time, group, target, raw target, and capacity columns
- `backbone.data`: file pattern, excluded files, and default folder
- `backbone.preprocessing`: drop columns, plateau/imputation, feature-engineering toggles
- `backbone.postprocess.clip_rules`: clipping and cleaning rules applied before split/train

This allows reuse of the same codebase as a generic time-series backbone by editing configuration instead of changing Python code.