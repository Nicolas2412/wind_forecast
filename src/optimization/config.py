from copy import deepcopy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.models import build_knn_model, build_tree_model
from tools import BACKBONE_CONFIG

_SCHEMA = BACKBONE_CONFIG.get("schema", {})
TIME_COL = _SCHEMA.get("time_column", "delivery_time")
TARGET_COL = _SCHEMA.get("target_column", "production_normalized")
SITE_COL = _SCHEMA.get("group_column", "site_name")
RANDOM_SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    builder: Callable[[dict[str, Any]], Any]
    default_params: dict[str, Any]
    suggest_params: Callable[["TrialLike"], dict[str, Any]]


class TrialLike(Protocol):
    def suggest_int(self, name: str, low: int, high: int, step: int | None = None) -> int:
        ...

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False,
    ) -> float:
        ...

    def suggest_categorical(self, name: str, choices: list[str]) -> str:
        ...


@dataclass(frozen=True)
class TabularConfig:
    model: str
    site: str
    train_percent: float
    n_trials: int
    mode: str
    data_folder: Path
    output_dir: Path
    time_col: str = TIME_COL
    target_col: str = TARGET_COL
    n_splits: int = 5


@dataclass(frozen=True)
class SequenceConfig:
    sequence_model: str
    seq_lengths: list[int]
    drop_prod_options: list[bool]
    site_start: int
    site_end: int
    test_size: float
    no_cv: bool
    output_dir: Path
    model_root: Path


@dataclass(frozen=True)
class SequenceExperiment:
    site_index: int
    model_type: str
    seq_len: int
    drop_prod: bool
    test_size: float
    no_cv: bool
    model_root: Path

    @property
    def run_name(self) -> str:
        suffix = "no_prod" if self.drop_prod else "with_prod"
        return f"{self.model_type}_seq{self.seq_len}_{suffix}"

    @property
    def savepath(self) -> Path:
        return self.model_root / self.model_type / f"site_{self.site_index}" / f"{self.run_name}.pkl"


def _rf_suggest(trial: TrialLike) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }


def _xgb_suggest(trial: TrialLike) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    }


def _lgbm_suggest(trial: TrialLike) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }


def _knn_suggest(trial: TrialLike) -> dict[str, Any]:
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 5, 80),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_int("p", 1, 2),
        "n_jobs": -1,
    }


TABULAR_MODELS: dict[str, ModelSpec] = {
    "random_forest": ModelSpec(
        builder=lambda params: build_tree_model("random_forest", params),
        default_params={
            "n_estimators": 200,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "max_depth": None,
        },
        suggest_params=_rf_suggest,
    ),
    "xgboost": ModelSpec(
        builder=lambda params: build_tree_model("xgboost", params),
        default_params={
            "n_estimators": 200,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "tree_method": "hist",
            "verbosity": 0,
            "learning_rate": 0.05,
            "max_depth": 6,
        },
        suggest_params=_xgb_suggest,
    ),
    "lightgbm": ModelSpec(
        builder=lambda params: build_tree_model("lightgbm", params),
        default_params={
            "n_estimators": 300,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
            "learning_rate": 0.05,
            "num_leaves": 63,
        },
        suggest_params=_lgbm_suggest,
    ),
    "knn": ModelSpec(
        builder=build_knn_model,
        default_params={
            "n_neighbors": 25,
            "weights": "distance",
            "n_jobs": -1,
        },
        suggest_params=_knn_suggest,
    ),
}

SUPPORTED_TABULAR = tuple(TABULAR_MODELS.keys())
SUPPORTED_SEQUENCE = ("lstm", "transformer")


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Integer list cannot be empty")
    return values


def parse_bool_list(raw: str) -> list[bool]:
    values: list[bool] = []
    for item in raw.split(","):
        value = item.strip().lower()
        if value in {"true", "1", "yes", "y"}:
            values.append(True)
        elif value in {"false", "0", "no", "n"}:
            values.append(False)
        elif value:
            raise ValueError(f"Invalid boolean value: {item}")
    if not values:
        raise ValueError("Boolean list cannot be empty")
    return values


def build_model(model_type: str, params: dict[str, Any]) -> Any:
    if model_type not in TABULAR_MODELS:
        raise ValueError(f"Unsupported model type: {model_type}")
    return TABULAR_MODELS[model_type].builder(params)


def default_params(model_type: str) -> dict[str, Any]:
    if model_type not in TABULAR_MODELS:
        raise ValueError(f"Unsupported model type: {model_type}")
    return deepcopy(TABULAR_MODELS[model_type].default_params)


def suggest_params(trial: TrialLike, model_type: str) -> dict[str, Any]:
    if model_type not in TABULAR_MODELS:
        raise ValueError(f"Unsupported model type: {model_type}")
    return TABULAR_MODELS[model_type].suggest_params(trial)


def resolve_tabular_models(selection: str) -> list[str]:
    if selection == "all":
        return list(SUPPORTED_TABULAR)
    if selection not in TABULAR_MODELS:
        raise ValueError(f"Unsupported model selection: {selection}")
    return [selection]


def resolve_sequence_models(selection: str) -> list[str]:
    if selection == "all":
        return list(SUPPORTED_SEQUENCE)
    if selection not in SUPPORTED_SEQUENCE:
        raise ValueError(f"Unsupported sequence model selection: {selection}")
    return [selection]
