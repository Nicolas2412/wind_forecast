from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.models import build_knn_model, build_lstm_net, build_transformer_net, build_tree_model

DEFAULT_TEST_SIZE = 0.2

SUPPORTED_MODELS = [
    "random_forest",
    "xgboost",
    "lightgbm",
    "knn",
    "lstm",
    "transformer",
]

DEFAULT_MODEL_PARAMS = {
    "random_forest": {
        "n_estimators": 200,
        "random_state": 42,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 200,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    },
    "lightgbm": {
        "n_estimators": 200,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "knn": {
        "n_neighbors": 25,
        "weights": "distance",
        "n_jobs": -1,
    },
    "lstm": {
        "seq_len": 48,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "epochs": 30,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "patience": 8,
        "grad_clip": 0.0,
        "val_fraction": 0.2,
    },
    "transformer": {
        "seq_len": 48,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "epochs": 30,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "patience": 8,
        "grad_clip": 0.0,
        "val_fraction": 0.2,
    },
}

DEFAULT_BACKBONE_CONFIG = {
    "data": {
        "folder": "data/",
        "file_pattern": "dataset_*.parquet",
        "exclude_files": ["dataset_2"],
    },
    "schema": {
        "time_column": "delivery_time",
        "group_column": "site_name",
        "target_column": "production_normalized",
        "raw_target_column": "production",
        "capacity_column": "installed_capacity",
    },
    "preprocessing": {
        "drop_columns": ["production", "installed_capacity", "is_not_plateau"],
        "drop_prod_columns": [
            "production_lag360h",
            "production_lag384h",
            "production_lag720h",
            "production_rolling_mean_7d",
            "production_rolling_std_7d",
            "production_rolling_mean_14d",
            "production_rolling_std_14d",
        ],
        "plateau": {
            "enabled": True,
            "N": 5,
            "window": "24h",
            "tolerance": 0.01,
            "low_thresh": 0.1,
            "high_thresh": 0.9,
        },
        "imputation": {
            "enabled": True,
            "max_gap_hours": 6,
        },
        "feature_engineering": {
            "time_features": True,
            "weather_features": True,
            "lag_features": True,
            "data_delay_days": 15,
        },
    },
    "postprocess": {
        "clip_rules": [
            {
                "contains_any": ["wind_speed", "wind_gusts"],
                "lower": 0.0,
                "upper": 45.0,
            },
            {
                "columns": ["theoretical_power"],
                "lower": 0.0,
                "upper": 5.0,
            },
            {
                "columns": ["wind_shear_alpha"],
                "replace_inf_with": 0.0,
                "fillna": 0.0,
                "lower": -0.5,
                "upper": 1.0,
            },
        ]
    },
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_config(config_path: str = "config.yaml") -> dict:
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path(__file__).resolve().parent / config_file

    if not config_file.exists():
        return {
            "valid_models": list(SUPPORTED_MODELS),
            "model_params": deepcopy(DEFAULT_MODEL_PARAMS),
            "backbone": deepcopy(DEFAULT_BACKBONE_CONFIG),
        }

    try:
        import yaml
    except ImportError:
        return {
            "valid_models": list(SUPPORTED_MODELS),
            "model_params": deepcopy(DEFAULT_MODEL_PARAMS),
            "backbone": deepcopy(DEFAULT_BACKBONE_CONFIG),
        }

    try:
        with config_file.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
    except Exception:
        raw = {}

    raw_valid_models = raw.get("valid_models", SUPPORTED_MODELS)
    valid_models = [name for name in raw_valid_models if name in SUPPORTED_MODELS]
    if not valid_models:
        valid_models = list(SUPPORTED_MODELS)

    model_params = deepcopy(DEFAULT_MODEL_PARAMS)
    loaded_params = raw.get("model_params", {})
    if isinstance(loaded_params, dict):
        for model_name, params in loaded_params.items():
            if model_name in model_params and isinstance(params, dict):
                model_params[model_name].update(params)

    backbone = deepcopy(DEFAULT_BACKBONE_CONFIG)
    loaded_backbone = raw.get("backbone", {})
    if isinstance(loaded_backbone, dict):
        _deep_update(backbone, loaded_backbone)

    return {
        "valid_models": valid_models,
        "model_params": model_params,
        "backbone": backbone,
    }


_APP_CONFIG = _load_config()
VALID_MODELS = _APP_CONFIG["valid_models"]
MODEL_PARAMS = _APP_CONFIG["model_params"]
BACKBONE_CONFIG = _APP_CONFIG["backbone"]
DEFAULT_MODEL_TYPE = VALID_MODELS[0]


def get_drop_columns(drop_prod: bool = False) -> list[str]:
    preprocessing = BACKBONE_CONFIG.get("preprocessing", {})
    columns = list(preprocessing.get("drop_columns", []))
    if drop_prod:
        columns.extend(preprocessing.get("drop_prod_columns", []))
    return list(dict.fromkeys(columns))


def apply_clip_rules(df: pd.DataFrame, rules: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    out = df.copy()
    active_rules = rules if rules is not None else BACKBONE_CONFIG.get("postprocess", {}).get("clip_rules", [])

    for rule in active_rules:
        if not isinstance(rule, dict):
            continue

        columns = list(rule.get("columns", []))
        if not columns:
            contains_any = list(rule.get("contains_any", []))
            if contains_any:
                columns = [
                    name for name in out.columns if any(fragment in name for fragment in contains_any)
                ]

        if not columns:
            continue

        existing = [name for name in columns if name in out.columns]
        if not existing:
            continue

        replace_inf_with = rule.get("replace_inf_with")
        if replace_inf_with is not None:
            out[existing] = out[existing].replace([float("inf"), float("-inf")], replace_inf_with)

        fillna_value = rule.get("fillna")
        if fillna_value is not None:
            out[existing] = out[existing].fillna(fillna_value)

        lower = rule.get("lower")
        upper = rule.get("upper")
        if lower is not None or upper is not None:
            out[existing] = out[existing].clip(lower=lower, upper=upper)

    return out


class DataProcessor:
    def __init__(self, path_folder: str, X: Optional[pd.DataFrame] = None, drop_columns: Optional[list] = None):
        self.path = path_folder
        schema = BACKBONE_CONFIG.get("schema", {})
        data_cfg = BACKBONE_CONFIG.get("data", {})
        preprocessing = BACKBONE_CONFIG.get("preprocessing", {})

        self.time_column = schema.get("time_column", "delivery_time")
        self.group_column = schema.get("group_column", "site_name")
        self.predict_column = schema.get("target_column", "production_normalized")
        self.raw_target_column = schema.get("raw_target_column", "production")
        self.capacity_column = schema.get("capacity_column", "installed_capacity")
        self.file_pattern = data_cfg.get("file_pattern", "dataset_*.parquet")
        self.exclude_files = set(data_cfg.get("exclude_files", []))

        dropped = drop_columns or []
        default_drop = preprocessing.get("drop_columns", ["production", "installed_capacity", "is_not_plateau"])
        self.drop_columns = list(dict.fromkeys([*dropped, *default_drop]))
        self.df = self.open_data() if X is None else X.copy()

    def open_data(self) -> pd.DataFrame:
        files = sorted(Path(self.path).glob(self.file_pattern))
        main_df = None

        for file in files:
            if any(fragment in file.name for fragment in self.exclude_files):
                continue
            current = pd.read_parquet(file)
            if self.time_column in current.columns:
                current[self.time_column] = pd.to_datetime(current[self.time_column], utc=True)

            if main_df is None:
                main_df = current
            else:
                if self.time_column in main_df.columns:
                    main_df[self.time_column] = pd.to_datetime(main_df[self.time_column], utc=True)
                main_df = pd.merge(
                    main_df,
                    current,
                    on=[self.group_column, self.time_column],
                    how="inner",
                )

        if main_df is None:
            return pd.DataFrame()
        return main_df

    def run(self) -> pd.DataFrame:
        preprocessing = BACKBONE_CONFIG.get("preprocessing", {})
        plateau_cfg = preprocessing.get("plateau", {})
        imputation_cfg = preprocessing.get("imputation", {})
        feature_cfg = preprocessing.get("feature_engineering", {})

        df = self.preprocess_data()
        if imputation_cfg.get("enabled", True):
            df = self.impute_production(df, max_gap_hours=int(imputation_cfg.get("max_gap_hours", 6)))
        df = self.engineer_features(
            df,
            data_delay_days=int(feature_cfg.get("data_delay_days", 15)),
            feature_cfg=feature_cfg,
        )
        return df

    def preprocess_data(
        self,
        N: int = 5,
        window: str = "24h",
        tolerance: float = 0.01,
        low_thresh: float = 0.1,
        high_thresh: float = 0.9,
    ) -> pd.DataFrame:
        df = self.df.copy()
        required = [self.group_column, self.time_column]
        if self.predict_column not in df.columns:
            required.extend([self.raw_target_column, self.capacity_column])
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if self.predict_column not in df.columns:
            df = df[df[self.capacity_column] != 0].copy()
            df[self.predict_column] = df[self.raw_target_column] / df[self.capacity_column]

        plateau_cfg = BACKBONE_CONFIG.get("preprocessing", {}).get("plateau", {})
        if plateau_cfg.get("enabled", True):
            df = compute_plateau(
                df=df,
                time_col=self.time_column,
                group_col=self.group_column,
                target_col=self.predict_column,
                N=int(plateau_cfg.get("N", N)),
                window=str(plateau_cfg.get("window", window)),
                tolerance=float(plateau_cfg.get("tolerance", tolerance)),
                low_thresh=float(plateau_cfg.get("low_thresh", low_thresh)),
                high_thresh=float(plateau_cfg.get("high_thresh", high_thresh)),
            )
        self.df = df
        return df

    def impute_production(self, df: pd.DataFrame, max_gap_hours: int = 6) -> pd.DataFrame:
        results = []

        for group_value, grp in df.groupby(self.group_column):
            local = grp.copy()
            local[self.time_column] = pd.to_datetime(local[self.time_column], utc=True)
            local = local.sort_values(self.time_column)

            if "is_not_plateau" in local.columns:
                local.loc[~local["is_not_plateau"], self.predict_column] = np.nan

            local = local.set_index(self.time_column)
            local[self.predict_column] = local[self.predict_column].interpolate(method="time", limit=max_gap_hours)
            local = local.reset_index()

            if local[self.predict_column].isna().any():
                local["tmp_hour"] = local[self.time_column].dt.hour
                hourly_map = local.groupby("tmp_hour")[self.predict_column].mean()
                hourly_map = hourly_map.fillna(local[self.predict_column].median())
                local[self.predict_column] = local[self.predict_column].fillna(local["tmp_hour"].map(hourly_map))
                local = local.drop(columns=["tmp_hour"])

            local[self.predict_column] = local[self.predict_column].ffill().bfill()
            local[self.group_column] = group_value
            results.append(local)

        if not results:
            return df

        return pd.concat(results, ignore_index=True)

    def engineer_features(
        self,
        df: pd.DataFrame,
        data_delay_days: int = 15,
        feature_cfg: Optional[dict[str, Any]] = None,
    ) -> pd.DataFrame:
        feature_cfg = feature_cfg or {}
        local = df.copy()
        local[self.time_column] = pd.to_datetime(local[self.time_column], utc=True)
        required_cols = [self.time_column, self.predict_column, self.group_column]
        if self.raw_target_column in local.columns and self.capacity_column in local.columns:
            required_cols.extend([self.raw_target_column, self.capacity_column])
        local = local.dropna(subset=required_cols)
        if self.capacity_column in local.columns:
            local = local[local[self.capacity_column] != 0]

        if feature_cfg.get("time_features", True):
            local["hour"] = local[self.time_column].dt.hour
            local["day_of_week"] = local[self.time_column].dt.dayofweek
            local["month"] = local[self.time_column].dt.month
            local["month_sin"] = np.sin(2 * np.pi * local["month"] / 12)
            local["month_cos"] = np.cos(2 * np.pi * local["month"] / 12)
            local["is_weekend"] = (local["day_of_week"] >= 5).astype(int)
            local["is_night"] = ((local["hour"] < 6) | (local["hour"] >= 22)).astype(int)
            local["hour_sin"] = np.sin(2 * np.pi * local["hour"] / 24)
            local["hour_cos"] = np.cos(2 * np.pi * local["hour"] / 24)
            local["dow_sin"] = np.sin(2 * np.pi * local["day_of_week"] / 7)
            local["dow_cos"] = np.cos(2 * np.pi * local["day_of_week"] / 7)

        if feature_cfg.get("weather_features", True) and "precipitation" in local.columns:
            local["precipitation"] = np.log1p(local["precipitation"].clip(lower=0))

        if self.raw_target_column in local.columns and self.capacity_column in local.columns:
            local[self.predict_column] = local[self.raw_target_column] / local[self.capacity_column]

        if feature_cfg.get("weather_features", True) and "wind_speed_100m" in local.columns and "wind_speed_10m" in local.columns:
            local["wind_speed_diff"] = local["wind_speed_100m"] - local["wind_speed_10m"]
            for col in ["wind_speed_10m", "wind_speed_100m"]:
                local[f"{col}_squared"] = local[col] ** 2
                local[f"{col}_cubed"] = local[col] ** 3

            local["wind_speed_ratio"] = local["wind_speed_100m"] / (local["wind_speed_10m"] + 1e-8)
            v10 = local["wind_speed_10m"].clip(lower=0.5)
            v100 = local["wind_speed_100m"].clip(lower=0.5)
            local["wind_shear_alpha"] = np.log(v100 / v10) / np.log(100 / 10)

        if feature_cfg.get("lag_features", True):
            min_lag_h = data_delay_days * 24
            results = []

            for group_value, grp in local.groupby(self.group_column):
                group_df = grp.set_index(self.time_column).sort_index()
                full_idx = pd.date_range(group_df.index.min(), group_df.index.max(), freq="1h", tz="UTC")
                group_df = group_df.reindex(full_idx)
                group_df[self.group_column] = group_value

                group_df[f"production_lag{min_lag_h}h"] = group_df[self.predict_column].shift(min_lag_h)
                group_df[f"production_lag{min_lag_h + 24}h"] = group_df[self.predict_column].shift(min_lag_h + 24)
                group_df["production_lag720h"] = group_df[self.predict_column].shift(720)

                shifted = group_df[self.predict_column].shift(min_lag_h)
                group_df["production_rolling_mean_7d"] = shifted.rolling(168, min_periods=84).mean()
                group_df["production_rolling_std_7d"] = shifted.rolling(168, min_periods=84).std()
                group_df["production_rolling_mean_14d"] = shifted.rolling(336, min_periods=168).mean()
                group_df["production_rolling_std_14d"] = shifted.rolling(336, min_periods=168).std()

                if "is_not_plateau" in group_df.columns:
                    group_df = group_df[group_df["is_not_plateau"].fillna(False)]

                results.append(group_df.reset_index().rename(columns={"index": self.time_column}))

            if not results:
                return local

            local = pd.concat(results, ignore_index=True)

        if "wind_direction_100m" in local.columns and "wind_speed_100m" in local.columns:
            rad = np.radians(local["wind_direction_100m"])
            local["wind_u_100m"] = -local["wind_speed_100m"] * np.sin(rad)
            local["wind_v_100m"] = -local["wind_speed_100m"] * np.cos(rad)

        if "wind_direction_10m" in local.columns and "wind_speed_10m" in local.columns:
            rad = np.radians(local["wind_direction_10m"])
            local["wind_u_10m"] = -local["wind_speed_10m"] * np.sin(rad)
            local["wind_v_10m"] = -local["wind_speed_10m"] * np.cos(rad)

        if "temperature_2m" in local.columns and "pressure_msl" in local.columns and "wind_speed_100m" in local.columns:
            local["air_density"] = local["pressure_msl"] / (287.05 * (local["temperature_2m"] + 273.15))
            local["theoretical_power"] = local["air_density"] * local["wind_speed_100m"] ** 3

        local = local.replace([np.inf, -np.inf], np.nan)
        return local

    def finalize_for_model(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_keep = [col for col in df.columns if col not in self.drop_columns]
        mask = df[cols_to_keep].notna().all(axis=1)
        return df.loc[mask, cols_to_keep].copy()


def compute_plateau(
    df: pd.DataFrame,
    time_col: str = "delivery_time",
    group_col: str = "site_name",
    target_col: str = "production_normalized",
    N: int = 5,
    window: str = "24h",
    tolerance: float = 0.01,
    low_thresh: float = 0.1,
    high_thresh: float = 0.9,
) -> pd.DataFrame:
    if target_col not in df.columns:
        out = df.copy()
        out["is_not_plateau"] = True
        return out

    def count_similar_in_window(series: pd.Series, time_window: str, tol: float) -> pd.Series:
        freq = series.index.to_series().diff().median()
        if pd.isna(freq) or freq == pd.Timedelta(0):
            freq = pd.Timedelta("1h")
        try:
            freq_delta = pd.to_timedelta(freq)
        except Exception:
            freq_delta = pd.Timedelta("1h")
        if freq_delta <= pd.Timedelta(0):
            freq_delta = pd.Timedelta("1h")
        half_window = pd.Timedelta(time_window) / 2
        ratio = half_window.total_seconds() / freq_delta.total_seconds()
        n_points = max(1, int(ratio))
        values = series.values
        counts = np.array(
            [
                np.sum(np.abs(values[max(0, i - n_points) : i + n_points + 1] - value) < tol)
                for i, value in enumerate(values)
            ]
        )
        return pd.Series(counts, index=series.index)

    outputs = []

    for _, site_df in df.groupby(group_col):
        local = site_df.sort_values(time_col).copy()
        local = local.set_index(time_col)

        p_max = local[target_col].max()
        p_low = p_max * low_thresh
        p_high = p_max * high_thresh

        local["similar_count"] = count_similar_in_window(local[target_col], window, tolerance)
        in_zone = (local[target_col] >= p_low) & (local[target_col] <= p_high)
        local["is_not_plateau"] = ~((local["similar_count"] >= N) & in_zone)

        outputs.append(local.drop(columns=["similar_count"]).reset_index())

    if not outputs:
        return df

    return pd.concat(outputs, ignore_index=True)


class ForecastModel:
    def __init__(
        self,
        model_type: str = DEFAULT_MODEL_TYPE,
        savepath: Optional[str] = None,
        verbose: bool = False,
        seq_len: int = 48,
    ):
        self.verbose = verbose
        self.model_type = self._validate_model_type(model_type)
        schema = BACKBONE_CONFIG.get("schema", {})
        self.time_column = schema.get("time_column", "delivery_time")
        self.group_column = schema.get("group_column", "site_name")
        self.predict_column = schema.get("target_column", "production_normalized")
        self.n_splits = 5
        self.savepath = savepath
        self.model: Any = None
        self.feature_columns = []
        self.input_size = None
        self.evaluation_results: dict[str, Any] = {}

        self.tree_defaults = {
            "random_forest": {
                "n_estimators": 200,
                "random_state": 42,
                "n_jobs": -1,
            },
            "xgboost": {
                "n_estimators": 200,
                "random_state": 42,
                "n_jobs": -1,
                "tree_method": "hist",
                "verbosity": 0,
            },
            "lightgbm": {
                "n_estimators": 200,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            },
        }
        self.knn_defaults = {
            "n_neighbors": 25,
            "weights": "distance",
            "n_jobs": -1,
        }
        self.lstm_params = {
            "seq_len": seq_len,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "epochs": 30,
            "batch_size": 256,
            "learning_rate": 1e-3,
        }
        self.transformer_params = {
            "seq_len": seq_len,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dropout": 0.1,
            "epochs": 30,
            "batch_size": 256,
            "learning_rate": 1e-3,
        }

        self.scaler_X = None
        self.scaler_y = None
        self._apply_config()

        if self.model_type in {"lstm", "transformer"}:
            self.scaler_X = StandardScaler()
            self.scaler_y = MinMaxScaler(feature_range=(0, 1))

        if self.savepath and Path(self.savepath).exists():
            self.load(self.savepath)

    def _validate_model_type(self, model_type: str) -> str:
        if model_type not in VALID_MODELS:
            raise ValueError(f"Invalid model type: {model_type}. Choose from {VALID_MODELS}")
        return model_type

    def _apply_config(self) -> None:
        params = MODEL_PARAMS.get(self.model_type, {})

        if self.model_type in self.tree_defaults:
            self.tree_defaults[self.model_type].update(params)
            return

        if self.model_type == "knn":
            self.knn_defaults.update(params)
            return

        if self.model_type == "lstm":
            self.lstm_params.update(params)
            return

        if self.model_type == "transformer":
            self.transformer_params.update(params)

    def train(self, df: pd.DataFrame, no_cv: bool = False) -> None:
        if self.model_type in {"random_forest", "xgboost", "lightgbm", "knn"}:
            self._train_tabular(df)
            return

        self._train_deep(df, no_cv=no_cv)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained or loaded")

        if self.model_type in {"random_forest", "xgboost", "lightgbm", "knn"}:
            return self._predict_tabular(df)
        pred, _, _, _ = self._predict_deep(df)
        return pred

    def evaluate(self, df: Optional[pd.DataFrame] = None, plot: bool = False) -> dict:
        results = dict(self.evaluation_results)

        if df is None:
            return results

        if self.model_type in {"lstm", "transformer"}:
            prediction, y_true, timestamps, sites = self._predict_deep(df)
        else:
            prediction = self.predict(df)
            y_true = df[self.predict_column].to_numpy()
            timestamps = pd.to_datetime(df[self.time_column], utc=True).to_numpy()
            if self.group_column in df.columns:
                sites = df[self.group_column].astype(str).to_numpy()
            else:
                sites = np.array(["global"] * len(df))

        if len(prediction) == 0:
            raise ValueError("No predictions were generated for evaluation")

        mae = mean_absolute_error(y_true, prediction)
        rmse = float(np.sqrt(mean_squared_error(y_true, prediction)))
        mean_y = float(np.mean(y_true)) if len(y_true) else 0.0
        nrmse = (rmse / mean_y) if mean_y != 0 else 0.0

        eval_frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(timestamps, utc=True),
                "group": sites,
                "y_true": y_true,
                "y_pred": prediction,
            }
        )

        per_group_metrics = {}
        for group_value, group_df in eval_frame.groupby("group"):
            group_mae = float(mean_absolute_error(group_df["y_true"], group_df["y_pred"]))
            group_rmse = float(np.sqrt(mean_squared_error(group_df["y_true"], group_df["y_pred"])))
            group_mean = float(group_df["y_true"].mean())
            group_nrmse = group_rmse / group_mean if group_mean != 0 else 0.0
            per_group_metrics[group_value] = {
                "mae": group_mae,
                "rmse": group_rmse,
                "nrmse": group_nrmse,
            }

        portfolio = eval_frame.groupby("timestamp", as_index=False)[["y_true", "y_pred"]].sum()
        n_groups = int(eval_frame["group"].nunique())

        portfolio_mae_total = float(mean_absolute_error(portfolio["y_true"], portfolio["y_pred"]))
        portfolio_rmse_total = float(np.sqrt(mean_squared_error(portfolio["y_true"], portfolio["y_pred"])))
        portfolio_mean = float(portfolio["y_true"].mean())
        portfolio_nrmse_total = portfolio_rmse_total / portfolio_mean if portfolio_mean != 0 else 0.0

        results.update(
            {
                "eval_mae": float(mae),
                "eval_rmse": float(rmse),
                "eval_nrmse": float(nrmse),
                "per_group_metrics": per_group_metrics,
                "per_site_metrics": per_group_metrics,
                "portfolio_mae_total": float(portfolio_mae_total),
                "portfolio_rmse_total": float(portfolio_rmse_total),
                "portfolio_nrmse_total": float(portfolio_nrmse_total),
                "portfolio_mae_per_group": float(portfolio_mae_total / n_groups if n_groups else 0.0),
                "portfolio_rmse_per_group": float(portfolio_rmse_total / n_groups if n_groups else 0.0),
                "portfolio_mae_per_site": float(portfolio_mae_total / n_groups if n_groups else 0.0),
                "portfolio_rmse_per_site": float(portfolio_rmse_total / n_groups if n_groups else 0.0),
                "n_groups": n_groups,
                "n_sites": n_groups,
            }
        )

        if plot:
            self._plot_evaluation(eval_frame)

        self.evaluation_results = results
        return results

    def save(self, path: Optional[str] = None) -> None:
        import joblib

        save_path = str(path or self.savepath)
        if not save_path:
            raise ValueError("No save path provided")
        if self.model is None:
            raise ValueError("No model available to save")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        if self.model_type in {"lstm", "transformer"}:
            import torch

            payload = {
                "model_state_dict": self.model.state_dict(),
                "feature_columns": self.feature_columns,
                "input_size": self.input_size,
                "scaler_X": self.scaler_X,
                "scaler_y": self.scaler_y,
            }
            torch.save(payload, save_path)
            return

        payload = {
            "model": self.model,
            "feature_columns": self.feature_columns,
        }
        joblib.dump(payload, save_path)

    def load(self, path: Optional[str] = None) -> None:
        import joblib

        load_path = str(path or self.savepath)
        if not load_path:
            raise ValueError("No load path provided")
        if not Path(load_path).exists():
            raise FileNotFoundError(load_path)

        if self.model_type in {"lstm", "transformer"}:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            payload = torch.load(load_path, map_location=device, weights_only=False)
            self.feature_columns = list(payload.get("feature_columns", []))
            self.input_size = payload.get("input_size")
            self.scaler_X = payload.get("scaler_X")
            self.scaler_y = payload.get("scaler_y")
            self.model = self._build_deep_model(self.input_size)
            self.model.load_state_dict(payload["model_state_dict"])
            self.model.to(device)
            self.model.eval()
            return

        payload = joblib.load(load_path)
        if isinstance(payload, dict) and "model" in payload:
            self.model = payload["model"]
            self.feature_columns = list(payload.get("feature_columns", []))
        else:
            self.model = payload
            self.feature_columns = []

    def _get_tabular_feature_columns(self, df: pd.DataFrame) -> list:
        return [
            col
            for col in df.columns
            if col not in {self.time_column, self.predict_column, self.group_column}
        ]

    def _build_tabular_model(self):
        if self.model_type in {"random_forest", "xgboost", "lightgbm"}:
            return build_tree_model(self.model_type, self.tree_defaults[self.model_type])
        if self.model_type == "knn":
            return build_knn_model(self.knn_defaults)
        raise ValueError(f"Unsupported tabular model type: {self.model_type}")

    def _train_tabular(self, df: pd.DataFrame) -> None:
        self.feature_columns = self._get_tabular_feature_columns(df)
        if not self.feature_columns:
            raise ValueError("No training features available")

        model = self._build_tabular_model()

        unique_times = np.sort(pd.to_datetime(df[self.time_column], utc=True).unique())
        splitter = TimeSeriesSplit(n_splits=self.n_splits)
        fold_maes = []

        for train_idx, val_idx in splitter.split(unique_times):
            train_dates = set(unique_times[train_idx])
            val_dates = set(unique_times[val_idx])

            train_df = df[pd.to_datetime(df[self.time_column], utc=True).isin(train_dates)]
            val_df = df[pd.to_datetime(df[self.time_column], utc=True).isin(val_dates)]

            if train_df.empty or val_df.empty:
                continue

            X_train = train_df[self.feature_columns]
            y_train = train_df[self.predict_column]
            X_val = val_df[self.feature_columns]
            y_val = val_df[self.predict_column]

            fold_model = (
                build_tree_model(self.model_type, self.tree_defaults[self.model_type])
                if self.model_type in self.tree_defaults
                else build_knn_model(self.knn_defaults)
            )
            fold_model.fit(X_train, y_train)
            pred = np.asarray(fold_model.predict(X_val))
            fold_maes.append(float(mean_absolute_error(y_val, pred)))

        X_full = df[self.feature_columns]
        y_full = df[self.predict_column]
        model.fit(X_full, y_full)
        self.model = model

        if fold_maes:
            self.evaluation_results = {
                "cv_mae": float(np.mean(fold_maes)),
                "cv_mae_std": float(np.std(fold_maes)),
            }
        else:
            self.evaluation_results = {
                "cv_mae": float("nan"),
                "cv_mae_std": float("nan"),
            }

    def _predict_tabular(self, df: pd.DataFrame) -> np.ndarray:
        if not self.feature_columns:
            self.feature_columns = self._get_tabular_feature_columns(df)
        X = df[self.feature_columns]
        return np.asarray(self.model.predict(X))

    def _get_deep_params(self) -> dict:
        if self.model_type == "lstm":
            return self.lstm_params
        return self.transformer_params

    def _build_deep_model(self, input_size: int):
        params = self._get_deep_params()
        if self.model_type == "lstm":
            return build_lstm_net(
                input_size=input_size,
                hidden_size=int(params["hidden_size"]),
                num_layers=int(params["num_layers"]),
                dropout=float(params["dropout"]),
            )
        return build_transformer_net(
            input_size=input_size,
            d_model=int(params["d_model"]),
            nhead=int(params["nhead"]),
            num_layers=int(params["num_layers"]),
            dropout=float(params["dropout"]),
        )

    def _prepare_deep_sequences(self, df: pd.DataFrame, fit_scalers: bool = False):
        params = self._get_deep_params()
        seq_len = int(params["seq_len"])

        local = df.copy()
        local[self.time_column] = pd.to_datetime(local[self.time_column], utc=True)

        if self.scaler_X is None or self.scaler_y is None:
            raise ValueError("Scalers are not initialized")

        required_cols = [self.time_column, self.predict_column, self.group_column] + self.feature_columns
        local = local.dropna(subset=required_cols)

        if fit_scalers:
            X_all = local[self.feature_columns].to_numpy(dtype=np.float32)
            y_all = local[[self.predict_column]].to_numpy(dtype=np.float32)
            self.scaler_X.fit(X_all)
            self.scaler_y.fit(y_all)

        all_X = []
        all_y = []
        all_time = []
        all_site = []

        for group_value, site_df in local.groupby(self.group_column):
            site_df = site_df.sort_values(self.time_column)
            if len(site_df) <= seq_len:
                continue

            X_site = site_df[self.feature_columns].to_numpy(dtype=np.float32)
            y_site = site_df[[self.predict_column]].to_numpy(dtype=np.float32)
            X_scaled = self.scaler_X.transform(X_site)
            y_scaled = self.scaler_y.transform(y_site).ravel()

            for idx in range(len(site_df) - seq_len):
                target_idx = idx + seq_len
                all_X.append(X_scaled[idx:target_idx])
                all_y.append(y_scaled[target_idx])
                all_time.append(site_df[self.time_column].iloc[target_idx])
                all_site.append(group_value)

        if not all_X:
            return np.array([]), np.array([]), np.array([]), np.array([])

        X_seq = np.asarray(all_X, dtype=np.float32)
        y_seq = np.asarray(all_y, dtype=np.float32)
        times = np.asarray(all_time)
        sites = np.asarray(all_site)
        return X_seq, y_seq, times, sites

    def _train_deep(self, df: pd.DataFrame, no_cv: bool = False) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.feature_columns = self._get_tabular_feature_columns(df)
        if not self.feature_columns:
            raise ValueError("No training features available")

        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))

        X_seq, y_seq, _, _ = self._prepare_deep_sequences(df, fit_scalers=True)
        if len(X_seq) == 0:
            raise ValueError("Not enough rows to create deep-learning sequences")

        self.input_size = int(X_seq.shape[2])
        self.model = self._build_deep_model(self.input_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        params = self._get_deep_params()
        epochs = int(params["epochs"])
        batch_size = int(params["batch_size"])
        learning_rate = float(params["learning_rate"])
        weight_decay = float(params.get("weight_decay", 0.0))
        patience = int(params.get("patience", 8))
        grad_clip = float(params.get("grad_clip", 0.0))
        val_fraction = float(params.get("val_fraction", 0.2))
        val_fraction = min(max(val_fraction, 0.05), 0.5)

        if no_cv or len(X_seq) < 20:
            X_train = X_seq
            y_train = y_seq
            X_val = np.array([])
            y_val = np.array([])
        else:
            val_size = max(1, int(val_fraction * len(X_seq)))
            X_train = X_seq[:-val_size]
            y_train = y_seq[:-val_size]
            X_val = X_seq[-val_size:]
            y_val = y_seq[-val_size:]

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=batch_size,
            shuffle=True,
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_state = None
        best_val_loss = float("inf")
        stalled = 0

        for _ in range(epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()

            if len(X_val) == 0:
                continue

            self.model.eval()
            with torch.no_grad():
                pred_val = self.model(torch.from_numpy(X_val).to(device)).cpu().numpy()
            val_loss = float(np.mean((pred_val - y_val) ** 2))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(self.model.state_dict())
                stalled = 0
            else:
                stalled += 1
                if stalled >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if len(X_val) > 0:
            self.model.eval()
            with torch.no_grad():
                pred_val = self.model(torch.from_numpy(X_val).to(device)).cpu().numpy()
            pred_val = self.scaler_y.inverse_transform(pred_val.reshape(-1, 1)).ravel()
            true_val = self.scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()
            mae = float(mean_absolute_error(true_val, pred_val))
            self.evaluation_results = {"cv_mae": mae, "cv_mae_std": 0.0}
        else:
            self.evaluation_results = {"cv_mae": float("nan"), "cv_mae_std": float("nan")}

    def _predict_deep(self, df: pd.DataFrame):
        import torch

        if self.model is None:
            raise ValueError("Model is not trained or loaded")
        if self.scaler_X is None or self.scaler_y is None:
            raise ValueError("Scalers are not available")

        X_seq, y_seq, times, sites = self._prepare_deep_sequences(df, fit_scalers=False)
        if len(X_seq) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        preds = []
        with torch.no_grad():
            for start in range(0, len(X_seq), 2048):
                end = start + 2048
                batch = torch.from_numpy(X_seq[start:end]).to(device)
                preds.append(self.model(batch).cpu().numpy())

        pred_scaled = np.concatenate(preds).reshape(-1, 1)
        pred = self.scaler_y.inverse_transform(pred_scaled).ravel()
        y_true = self.scaler_y.inverse_transform(y_seq.reshape(-1, 1)).ravel()
        return pred, y_true, times, sites

    def _plot_evaluation(self, eval_frame: pd.DataFrame) -> None:
        import matplotlib.pyplot as plt

        plot_df = eval_frame.sort_values("timestamp")
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(plot_df["timestamp"], plot_df["y_true"], label="actual", linewidth=1)
        ax.plot(plot_df["timestamp"], plot_df["y_pred"], label="predicted", linewidth=1)
        ax.legend()
        ax.set_title(f"Evaluation - {self.model_type}")
        plt.tight_layout()
        plt.show()
