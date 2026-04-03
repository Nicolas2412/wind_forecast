import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from tools import DataProcessor

from .config import SITE_COL, TabularConfig, build_model, default_params, resolve_tabular_models, suggest_params


@dataclass
class TabularTrainResult:
    model: Any
    metrics: dict[str, Any]
    features: list[str]


def prepare_xy(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    drop_columns = [name for name in (time_col, target_col, SITE_COL) if name in df.columns]
    features = df.drop(columns=drop_columns)
    target = df[target_col]
    return features, target


def split_time(df: pd.DataFrame, train_percent: float, time_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = df.copy()
    local[time_col] = pd.to_datetime(local[time_col], utc=True)
    start_time = local[time_col].min()
    end_time = local[time_col].max()
    threshold = start_time + (end_time - start_time) * train_percent
    train_df = local[local[time_col] < threshold].copy()
    test_df = local[local[time_col] >= threshold].copy()
    return train_df, test_df


def make_time_folds(
    df: pd.DataFrame,
    time_col: str,
    n_splits: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    local = df.copy()
    local[time_col] = pd.to_datetime(local[time_col], utc=True)
    unique_times = np.sort(local[time_col].unique())

    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    for train_idx, val_idx in splitter.split(unique_times):
        train_times = set(unique_times[train_idx])
        val_times = set(unique_times[val_idx])

        train_df = local[local[time_col].isin(train_times)]
        val_df = local[local[time_col].isin(val_times)]

        if not train_df.empty and not val_df.empty:
            folds.append((train_df, val_df))

    return folds


def cross_validate(
    model_type: str,
    params: dict[str, Any],
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    time_col: str,
    target_col: str,
) -> float:
    maes: list[float] = []

    for train_df, val_df in folds:
        X_train, y_train = prepare_xy(train_df, time_col=time_col, target_col=target_col)
        X_val, y_val = prepare_xy(val_df, time_col=time_col, target_col=target_col)

        model = build_model(model_type, params)
        model.fit(X_train, y_train)
        predictions = np.asarray(model.predict(X_val))
        maes.append(float(mean_absolute_error(y_val, predictions)))

    return float(np.mean(maes)) if maes else float("inf")


def tune_model(
    model_type: str,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    n_trials: int,
    time_col: str,
    target_col: str,
) -> dict[str, Any]:
    optuna = importlib.import_module("optuna")

    def objective(trial):
        params = suggest_params(trial, model_type)
        return cross_validate(model_type, params, folds, time_col=time_col, target_col=target_col)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return dict(study.best_params)


def save_artifacts(
    model_type: str,
    model: Any,
    metrics: dict[str, Any],
    feature_names: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / f"{model_type}_model.joblib")
    (output_dir / f"{model_type}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / f"{model_type}_features.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")


class TabularOptimizer:
    def __init__(
        self,
        config: TabularConfig,
        processor: DataProcessor | None = None,
        logger: logging.Logger | None = None,
    ):
        self.config = config
        self.processor = processor or DataProcessor(str(config.data_folder))
        self.logger = logger or logging.getLogger(__name__)

    def load_data(self) -> pd.DataFrame:
        raw = self.processor.run()
        if self.config.site != "all":
            raw = raw[raw[SITE_COL] == self.config.site].copy()
            if raw.empty:
                raise ValueError(f"No rows available for site '{self.config.site}'")
        return self.processor.finalize_for_model(raw)

    def train_and_evaluate(
        self,
        model_type: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        folds: list[tuple[pd.DataFrame, pd.DataFrame]],
        params: dict[str, Any],
    ) -> TabularTrainResult:
        X_train, y_train = prepare_xy(train_df, self.config.time_col, self.config.target_col)
        X_test, y_test = prepare_xy(test_df, self.config.time_col, self.config.target_col)

        model = build_model(model_type, params)
        model.fit(X_train, y_train)
        predictions = np.asarray(model.predict(X_test))

        metrics = {
            "cv_mae": cross_validate(
                model_type,
                params,
                folds,
                time_col=self.config.time_col,
                target_col=self.config.target_col,
            ),
            "test_mae": float(mean_absolute_error(y_test, predictions)),
            "params": params,
        }

        return TabularTrainResult(model=model, metrics=metrics, features=list(X_train.columns))

    def _resolve_params(
        self,
        model_type: str,
        output_dir: Path,
        folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    ) -> dict[str, Any] | None:
        best_path = output_dir / f"{model_type}_best_params.json"

        if self.config.mode in {"tune", "all"}:
            self.logger.info("Tuning %s with %d trials", model_type, self.config.n_trials)
            params = tune_model(
                model_type,
                folds,
                self.config.n_trials,
                time_col=self.config.time_col,
                target_col=self.config.target_col,
            )
            best_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
            return params

        if best_path.exists():
            return json.loads(best_path.read_text(encoding="utf-8"))

        return default_params(model_type)

    def run(self) -> dict[str, dict[str, Any]]:
        dataset = self.load_data()
        train_df, test_df = split_time(dataset, self.config.train_percent, self.config.time_col)
        folds = make_time_folds(train_df, self.config.time_col, self.config.n_splits)

        output_dir = self.config.output_dir / "tabular_optimization"
        output_dir.mkdir(parents=True, exist_ok=True)

        selected_models = resolve_tabular_models(self.config.model)
        all_metrics: dict[str, dict[str, Any]] = {}

        for model_type in selected_models:
            params = self._resolve_params(model_type, output_dir, folds)
            if params is None or self.config.mode == "tune":
                continue

            result = self.train_and_evaluate(model_type, train_df, test_df, folds, params)
            save_artifacts(model_type, result.model, result.metrics, result.features, output_dir)
            all_metrics[model_type] = result.metrics

        if all_metrics:
            summary_path = output_dir / f"summary_{self.config.model}.json"
            summary_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

        return all_metrics
