# Wind Power Forecasting Pipeline

This project implements a complete machine learning and deep learning pipeline to forecast wind power production across multiple highly-correlated wind farm sites. It features automatic data preprocessing, plateau detection, causal feature engineering, and robust time-series validation.

## Supported Models
- **Deep Learning**: LSTM, Transformer
- **Tree-based**: Random Forest, XGBoost, LightGBM
- **Statistical**: SARIMAX

## Features
- **Temporal Alignment**: Combines multiple parquet datasets (`dataset_1`, etc.) precisely on `delivery_time` and `site_name`.
- **Wind Farm Physics Features**: Calculates `air_density`, `wind_shear_alpha`, `theoretical_power`, and splits wind vectors ($U, V$).
- **Causal Auto-regressive Lags**: Computes trailing values and rolling statistics strictly avoiding future-leakage (min lag: 15 days).
- **Plateau Detection**: Automatically identifies periods of clipped generation (maintenance or grid curtailment) to avoid poisoning the model.
- **Robust Cross-Validation**: Implements true `TimeSeriesSplit` across non-overlapping temporal windows to validate sequential predictions.

## Setup

First, place your raw parquet files inside a `data/` folder at the root of the project.

### Configuration
Hyperparameters are managed centrally.
- You can override default model settings using `config.yaml`.
- Deep learning defaults can be found directly in `tools.py` (or uncomment the config loader line to pull them from yaml).

## Usage

You interact with the pipeline via `main.py`.

```bash
# Run a Transformer model training and evaluation using all sites
python main.py -m transformer -n my_first_transformer

# Run using a specific model (e.g. lighbm)
python main.py -m lightgbm -n my_lgbm_model

# Run with a smaller test set proportion (e.g. 10%)
python main.py -m xgboost -ts 0.1

# Train and evaluate on ONE site only (by default site index 0)
python main.py -m lstm -u -s 0 -n single_site_lstm

# Drop production-based lagging features (useful for 0-shot forecasting)
python main.py -m random_forest -dp

# Skip Cross-Validation during training (trains directly on all train data)
python main.py -m lstm --no_cv

# Skip training entirely and load an existing model to evaluate
python main.py -m transformer --skip_train -n my_first_transformer
```

### Command-Line Arguments Reference
- `-m`, `--model`: The model to use (`random_forest`, `xgboost`, `lightgbm`, `sarimax`, `lstm`, `transformer`). Default: `random_forest`.
- `-ts`, `--test_size`: Proportion of the timeline to use for validation. Default: `0.2` (20%).
- `-u`, `--unique_site`: Train and test on a single site instead of the whole portfolio.
- `-s`, `--site_index`: Which site index to pick if `-u` is active. Default: `0`.
- `-dp`, `--drop_prod`: Disables production-based historical features (lags, rolling means/stds).
- `--no_cv`: Skips the time-series cross validation phase for Deep Learning models.
- `--skip_train`: Skips training and attempts to load `[model_type]/[name].pkl` for immediate evaluation.
- `-n`, `--name`: Name of the model file to save/load. Default: `testModel`.

### Outputs
Models are saved in `data/models/{model_type}/{subfolder}/`.
During evaluation, the pipeline plots per-site MAE/RMSE trajectories and complete portfolio (summed) metrics. These evaluation graphs are saved into the corresponding model's `evaluation` subdirectory.
