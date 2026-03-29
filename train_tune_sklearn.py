"""
train_tune_sklearn.py
-----------------
Entraîne et tune les hyperparamètres de Random Forest, LightGBM et XGBoost
sur des données de production de parcs éoliens offshore (day-ahead J+1).

Utilisation
-----------
    python train_tune_sklearn.py --data_folder data/ --mode tune
    python train_tune_sklearn.py --data_folder data/ --mode train --model lightgbm
    python train_tune_sklearn.py --data_folder data/ --mode all

Arguments
---------
    --data_folder   Chemin vers le dossier contenant les fichiers .parquet
    --mode          "tune"  → Optuna, affiche les meilleurs params
                    "train" → Entraîne avec les params recommandés ou précédemment tunés
                    "all"   → Tune puis réentraîne
    --model         Modèle(s) ciblé(s) — s'applique à tous les modes :
                    random_forest | lightgbm | xgboost | all  (défaut : all)
    --train_percent Pourcentage du dataset à inclure dans l'ensemble de train (80% par défaut)
    --n_trials      Nombre d'essais Optuna par modèle (défaut : 50)
    --output_dir    Dossier de sortie pour les modèles sérialisés (défaut : models/)
    --site          Nom d'un site pour créer un modèle spécialisé sur un site (défaut : all)
                    all_indiv permet l'entraînement de modèles spécialisé pour tous les sites
"""

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
import os

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
from tqdm.auto import trange

from tools import DataProcessor, ForecastModel

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

N_SPLITS = 5          # folds TimeSeriesSplit
EARLY_STOPPING = 50   # rounds sans amélioration (LightGBM / XGBoost)
RANDOM_STATE = 42
SITE_LIST = ['Norther Offshore WP', 'Northwind', 'Belwind Phase 1', 'Northwester 2',
            'Thorntonbank - C-Power - Area NE', 'Mermaid Offshore WP',
            'Seastar Offshore WP', 'Nobelwind Offshore Windpark',
            'Thorntonbank - C-Power - Area SW', 'Rentel Offshore WP']


# ---------------------------------------------------------------------------
# Correctif : feature engineering robuste aux trous temporels
# ---------------------------------------------------------------------------

def engineer_features_dayahead(
    df: pd.DataFrame,
    time_col: str = "delivery_time",
    data_delay_days: int = 15,
) -> pd.DataFrame:
    """
    Feature engineering pour la prévision day-ahead J+1 avec un délai
    de réception des données de production de `data_delay_days` jours.

    Contrainte de causalité stricte
    --------------------------------
    Au moment de la prévision pour l'heure H, la dernière donnée de
    production disponible date de H - (data_delay_days * 24) heures.
    Tout lag inférieur à ce seuil est du data leakage et est supprimé.

    Avec data_delay_days=15 :
      - lag_24h   → leakage  ❌  (J-1, non reçu)
      - lag_48h   → leakage  ❌  (J-2, non reçu)
      - lag_360h  → safe     ✅  (J-15, premier lag utilisable)
      - lag_384h  → safe     ✅  (J-16)
      - lag_720h  → safe     ✅  (J-30, même heure 30 jours avant)
      - rolling sur [J-15, J-22]  → safe ✅
      - rolling sur [J-15, J-29]  → safe ✅

    Features météo NWP (vent, température, pression…) : toujours
    disponibles car issues de prévisions numériques.

    Ajoute également :
      - composantes vectorielles du vent (u, v) à 100 m
      - densité de l'air (approx. gaz parfait)
      - puissance théorique ∝ ρ·v³
    """
    # Premier lag utilisable en heures
    min_lag_h = data_delay_days * 24   # 360h pour 15 jours

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    results = []
    for site, grp in df.groupby("site_name"):
        grp = grp.set_index(time_col).sort_index()

        # --- Grille horaire complète (corrige les trous de maintenance) ---
        full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq="1h", tz="UTC")
        grp = grp.reindex(full_idx)
        grp["site_name"] = site

        # --- Lags causaux (>= data_delay_days * 24h) ---
        # lag_360h : même heure, J-15 — premier lag disponible
        grp[f"production_lag{min_lag_h}h"] = grp["production_normalized"].shift(min_lag_h)
        # lag_384h : même heure, J-16
        grp[f"production_lag{min_lag_h + 24}h"] = grp["production_normalized"].shift(min_lag_h + 24)
        # lag_720h : même heure, J-30 (capture la saisonnalité mensuelle)
        grp["production_lag720h"] = grp["production_normalized"].shift(720)

        # --- Statistiques glissantes causales ---
        # On shifte d'abord de min_lag_h pour que la fenêtre commence à J-15
        shifted = grp["production_normalized"].shift(min_lag_h)
        # Fenêtre 7 jours (168h) sur [J-15, J-22]
        grp["production_rolling_mean_7d"] = shifted.rolling(168, min_periods=84).mean()
        grp["production_rolling_std_7d"]  = shifted.rolling(168, min_periods=84).std()
        # Fenêtre 14 jours (336h) sur [J-15, J-29]
        grp["production_rolling_mean_14d"] = shifted.rolling(336, min_periods=168).mean()
        grp["production_rolling_std_14d"]  = shifted.rolling(336, min_periods=168).std()

        # --- Suppression des lignes de maintenance APRÈS calcul des lags ---
        if "is_not_plateau" in grp.columns:
            grp = grp[grp["is_not_plateau"].fillna(False)]

        results.append(grp.reset_index().rename(columns={"index": time_col}))

    df = pd.concat(results, ignore_index=True)

    # --- Features physiques vent (issues du NWP, toujours disponibles) ---
    if "wind_direction_100m" in df.columns:
        rad = np.radians(df["wind_direction_100m"])
        # Convention météo : direction = origine du vent (0°=Nord, 90°=Est)
        # U (zonal, positif vers l'Est)     = -speed * sin(dir)
        # V (méridional, positif vers le Nord) = -speed * cos(dir)
        df["wind_u_100m"] = -df["wind_speed_100m"] * np.sin(rad)
        df["wind_v_100m"] = -df["wind_speed_100m"] * np.cos(rad)

    if "temperature_2m" in df.columns and "pressure_msl" in df.columns:
        # Densité de l'air : ρ = P / (R_d * T),  R_d = 287.05 J/(kg·K)
        df["air_density"] = df["pressure_msl"] / (287.05 * (df["temperature_2m"] + 273.15))
        # Puissance éolienne théorique ∝ ρ·v³ (avant coefficient de puissance Cp)
        df["theoretical_power"] = df["air_density"] * df["wind_speed_100m"] ** 3

    # --- Nettoyage des colonnes intermédiaires et anciens lags leakés ---
    drop_cols = ["is_not_plateau", "similar_count",
                 "production_lag1", "production_lag24", "production_lag48"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["production_normalized"], inplace=True)


    return df


# ---------------------------------------------------------------------------
# Préparation X / y pour les modèles sklearn-like
# ---------------------------------------------------------------------------

def prepare_Xy(
    df: pd.DataFrame,
    time_col: str = "delivery_time",
    target: str = "production_normalized",
):
    """Renvoie (X, y) en supprimant les colonnes non-features."""
    drop = [c for c in [time_col, target, "site_name"] if c in df.columns]
    X = df.drop(columns=drop)
    y = df[target]
    return X, y


# ---------------------------------------------------------------------------
# Construction des modèles avec paramètres personnalisés
# ---------------------------------------------------------------------------

def build_model(model_type: str, params: dict):
    """Instancie le modèle sklearn/lgbm/xgb avec les params fournis."""
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)

    if model_type == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            **params,
        )

    if model_type == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
            **params,
        )

    raise ValueError(f"Modèle inconnu : {model_type}")


# ---------------------------------------------------------------------------
# Évaluation CV avec early stopping optionnel
# ---------------------------------------------------------------------------

def cross_validate(
    model_type: str,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    desc: str = "",
) -> tuple[float, float]:
    """
    TimeSeriesSplit CV. Renvoie (mae_mean, mae_std).
    Active l'early stopping pour LightGBM et XGBoost.
    Affiche une barre de progression par fold avec la MAE courante.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    label = desc or f"  [{model_type}] CV"

    fold_bar = tqdm(
        tscv.split(X),
        total=n_splits,
        desc=label,
        unit="fold",
        leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} folds [{elapsed}<{remaining}] {postfix}",
    )

    for train_idx, val_idx in fold_bar:
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = build_model(model_type, params)

        if model_type == "lightgbm":
            from lightgbm import early_stopping as lgb_es, log_evaluation
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb_es(EARLY_STOPPING, verbose=False), log_evaluation(-1)],
            )

        elif model_type == "xgboost":
            model.set_params(early_stopping_rounds=EARLY_STOPPING)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        else:
            model.fit(X_tr, y_tr)

        fold_mae = mean_absolute_error(y_val, model.predict(X_val))
        maes.append(fold_mae)
        fold_bar.set_postfix({"MAE fold": f"{fold_mae:.4f}", "MAE moy.": f"{np.mean(maes):.4f}"})

    return float(np.mean(maes)), float(np.std(maes))


# ---------------------------------------------------------------------------
# Espaces de recherche Optuna
# ---------------------------------------------------------------------------

def suggest_params(trial: optuna.Trial, model_type: str) -> dict:
    """Définit l'espace de recherche par modèle."""

    if model_type == "random_forest":
        return {
            "n_estimators":    trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth":       trial.suggest_int("max_depth", 8, 30),
            "min_samples_leaf":trial.suggest_int("min_samples_leaf", 10, 100),
            "max_features":    trial.suggest_float("max_features", 0.3, 0.9),
            "max_samples":     trial.suggest_float("max_samples", 0.6, 1.0),
        }

    if model_type == "lightgbm":
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 3000, step=50),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "objective":         "regression_l1",  # MAE directe
        }

    if model_type == "xgboost":
        return {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 3000, step=50),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":       trial.suggest_int("max_depth", 3, 10),
            "min_child_weight":trial.suggest_int("min_child_weight", 1, 50),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "objective":       "reg:absoluteerror",  # MAE directe
        }

    raise ValueError(f"Modèle inconnu : {model_type}")


# ---------------------------------------------------------------------------
# Lancement d'une étude Optuna
# ---------------------------------------------------------------------------

def tune_model(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
) -> tuple[dict, float]:
    """
    Lance une étude Optuna pour `model_type`.
    Renvoie (best_params, best_mae).
    Affiche une barre de progression par essai avec la meilleure MAE courante.
    """
    log.info(f"[{model_type}] Démarrage du tuning — {n_trials} essais")

    trial_bar = tqdm(
        total=n_trials,
        desc=f"[{model_type}] Tuning",
        unit="essai",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} essais [{elapsed}<{remaining}] {postfix}",
    )
    trial_bar.set_postfix({"meilleure MAE": "—"})

    def objective(trial):
        params = suggest_params(trial, model_type)
        mae, _ = cross_validate(
            model_type, params, X, y,
            desc=f"  [{model_type}] essai {trial.number + 1:>{len(str(n_trials))}}",
        )
        try:
            best_so_far = trial.study.best_value
        except ValueError:
            best_so_far = mae
        trial_bar.set_postfix({"meilleure MAE": f"{min(mae, best_so_far):.4f}", "MAE essai": f"{mae:.4f}"})
        trial_bar.update(1)
        return mae

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    trial_bar.close()

    best = study.best_trial
    log.info(
        f"[{model_type}] Tuning terminé — meilleur essai #{best.number} — "
        f"MAE CV = {best.value:.4f}"
    )
    return best.params, best.value


# ---------------------------------------------------------------------------
# Entraînement final sur tout le jeu d'entraînement
# ---------------------------------------------------------------------------

def train_final(
    model_type: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> tuple:
    """
    1. CV pour reporter les métriques officielles.
    2. Réentraîne sur X_train complet.
    3. Évalue sur X_eval (hold-out temporel).
    Renvoie (model, metrics_dict).
    """
    log.info(f"[{model_type}] CV finale avec les meilleurs paramètres...")
    t0 = time.time()
    cv_mae, cv_std = cross_validate(
        model_type, params, X_train, y_train,
        desc=f"[{model_type}] CV finale",
    )
    log.info(f"[{model_type}] CV MAE = {cv_mae:.4f} ± {cv_std:.4f}  ({time.time()-t0:.0f}s)")

    # Réentraînement complet
    log.info(f"[{model_type}] Entraînement final sur l'ensemble du train...")
    model = build_model(model_type, params)
    t1 = time.time()

    if model_type == "lightgbm":
        from lightgbm import early_stopping as lgb_es, log_evaluation
        # Callback tqdm pour LightGBM : avancement par round
        # Récupérer le n_estimators qui sera réellement utilisé

        n_est = params.get("n_estimators", 1000)
        pbar = tqdm(total=n_est, desc=f"[{model_type}] Fit final", unit="round", leave=True)
        rounds_done = [0]

        class _TqdmCallback:
            def __call__(self, env):
                delta = env.iteration - rounds_done[0]
                pbar.update(delta)
                rounds_done[0] = env.iteration
                best = env.evaluation_result_list[0][2] if env.evaluation_result_list else None
                if best:
                    pbar.set_postfix({"MAE val": f"{best:.4f}", "round": env.iteration})
            def before_iteration(self, env): return False

        model.fit(
            X_train, y_train,
            eval_set=[(X_eval, y_eval)],
            callbacks=[
                lgb_es(EARLY_STOPPING, verbose=False),
                log_evaluation(-1),
                _TqdmCallback(),
            ],
        )
        pbar.n = rounds_done[0]
        pbar.close()

    elif model_type == "xgboost":
        from xgboost.callback import TrainingCallback
        # Récupérer le n_estimators qui sera réellement utilisé
        n_est = params.get("n_estimators", 1000)

        pbar = tqdm(total=n_est, desc=f"[{model_type}] Fit final", unit="round", leave=True)

        class _XGBTqdm(TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                pbar.update(1)
                if evals_log:
                    metric_vals = list(list(evals_log.values())[-1].values())[-1]
                    if metric_vals:
                        pbar.set_postfix({"MAE val": f"{metric_vals[-1]:.4f}", "round": epoch})
                return False
        model.set_params(early_stopping_rounds=EARLY_STOPPING, callbacks=[_XGBTqdm()])
        model.fit(
            X_train, y_train,
            eval_set=[(X_eval, y_eval)],
            verbose=False,
        )
        pbar.close()

    else:
        # Random Forest : pas de rounds, on affiche juste un spinner
        with tqdm(total=1, desc=f"[{model_type}] Fit final", unit="modèle", leave=True) as pbar:
            model.fit(X_train, y_train)
            pbar.update(1)

    log.info(f"[{model_type}] Fit terminé ({time.time()-t1:.0f}s)")

    # Évaluation hold-out
    y_pred = model.predict(X_eval)
    test_mae = float(mean_absolute_error(y_eval, y_pred))
    log.info(f"[{model_type}] MAE hold-out = {test_mae:.4f}")

    metrics = {
        "cv_mae":      round(cv_mae, 6),
        "cv_std":      round(cv_std, 6),
        "test_mae":    round(test_mae, 6),
        "best_params": params,
    }
    return model, metrics


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

def save_artifacts(
    model_type: str,
    model,
    metrics: dict,
    feature_names: list,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path   = output_dir / f"{model_type}_model.joblib"
    metrics_path = output_dir / f"{model_type}_metrics.json"
    features_path = output_dir / f"{model_type}_features.json"
    
    if model_type == "xgboost" and hasattr(model, "set_params"):
        model.set_params(callbacks=None)

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    features_path.write_text(json.dumps(feature_names, indent=2))

    log.info(f"[{model_type}] Modèle sauvegardé → {model_path}")
    log.info(f"[{model_type}] Métriques       → {metrics_path}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline(args):
    t_global = time.time()
    n_models = len(["random_forest", "lightgbm", "xgboost"] if args.model == "all" else [args.model])

    # Étapes du pipeline affichées comme une barre globale
    pipeline_steps = ["Chargement données", "Feature engineering", "Split train/eval"]
    if args.mode in ("tune", "all"):
        pipeline_steps += [f"Tuning {m}" for m in (["random_forest", "lightgbm", "xgboost"] if args.model == "all" else [args.model])]
    if args.mode in ("train", "all"):
        pipeline_steps += [f"Entraînement {m}" for m in (["random_forest", "lightgbm", "xgboost"] if args.model == "all" else [args.model])]

    pipeline_bar = tqdm(
        total=len(pipeline_steps),
        desc="Pipeline",
        unit="étape",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} étapes [{elapsed}] {postfix}",
        position=0,
    )

    def step(name):
        pipeline_bar.set_postfix({"étape": name})
        return name

    # --- 1. Chargement et preprocessing via DataProcessor ---
    step("Chargement données")
    log.info("Chargement des données...")
    processor = DataProcessor(args.data_folder)
    df = processor.prepocess_data()
    pipeline_bar.update(1)

    # Feature engineering corrigé pour day-ahead
    step("Feature engineering")
    log.info("Feature engineering day-ahead (grille horaire régulière)...")
    df = engineer_features_dayahead(df)
    if args.site != "all":
        log.info(f"Filtrage du dataset pour le site : {args.site}")
        df = df[df["site_name"] == args.site].copy()

        if df.empty:
            raise ValueError(f"Aucune donnée trouvée pour le site : {args.site}")
    drop_always = ["production", "installed_capacity"]
    df.drop(columns=[c for c in drop_always if c in df.columns], inplace=True)
    log.info(f"Dataset prêt : {len(df):,} lignes × {df.shape[1]} colonnes")
    pipeline_bar.update(1)

    # --- 2. Split temporel train / eval ---
    step("Split train/eval")
    time_col = processor.time_column
    t_min = df[time_col].min()
    t_max = df[time_col].max()
    total_duration = t_max - t_min

    # Derniers 20% du temps → test, reste → train/val en CV
    t_test_start = t_min + total_duration * args.train_percent

    df_train = df[df[time_col] < t_test_start].copy()
    df_test = df[df[time_col] >= t_test_start].copy()

    print(
        f"Période train : {df_train[time_col].min()}  →  {df_train[time_col].max()}")
    print(
        f"Période test  : {df_test[time_col].min()}  →  {df_test[time_col].max()}")

    
    X_train, y_train = prepare_Xy(df_train)
    X_eval,  y_eval  = prepare_Xy(df_test)
    feature_names = list(X_train.columns)
    log.info(
        f"Train : {len(df_train):,} lignes | "
        f"Eval  : {len(df_test):,} lignes | "
        f"Features : {len(feature_names)}"
    )
    pipeline_bar.update(1)

    ALL_MODELS = ["random_forest", "lightgbm", "xgboost"]
    models_to_run = ALL_MODELS if args.model == "all" else [args.model]
    output_dir = Path(args.output_dir)

    # --- 3. Paramètres par défaut (recommandés) ---
    DEFAULT_PARAMS = {
        "random_forest": {
            "n_estimators":     100,
            "max_depth":        None,
            "min_samples_leaf": 50,
            "max_features":     0.7,
        },
        "lightgbm": {
            "n_estimators":      400,
            "learning_rate":     0.05,
            "num_leaves":        63,
            "min_child_samples": 50,
            "subsample":         0.8,
            "colsample_bytree":  0.7,
            "reg_alpha":         0.1,
            "reg_lambda":        1.0,
            "objective":         "regression_l1",
        },
        "xgboost": {
            "n_estimators":     100,
            "learning_rate":    0.05,
            "max_depth":        6,
            "min_child_weight": 5,
            "subsample":        0.8,
            "colsample_bytree": 0.7,
            "reg_alpha":        0.1,
            "reg_lambda":       1.0,
            "objective":        "reg:absoluteerror",
        },
    }

    all_metrics = {}

    # --- 4. Mode TUNE ---
    if args.mode in ("tune", "all"):
        log.info("=" * 60)
        log.info(f"MODE TUNING — Optuna ({len(models_to_run)} modèle(s), {args.n_trials} essais chacun)")
        log.info("=" * 60)

        best_params_all = {}
        for mt in models_to_run:
            step(f"Tuning {mt}")
            best_params, best_mae = tune_model(mt, X_train, y_train, n_trials=args.n_trials)
            best_params_all[mt] = best_params
            params_path = output_dir / f"{mt}_best_params.json"
            output_dir.mkdir(parents=True, exist_ok=True)
            params_path.write_text(json.dumps(best_params, indent=2))
            log.info(f"[{mt}] Params sauvegardés → {params_path}")
            pipeline_bar.update(1)

        if args.mode == "tune":
            pipeline_bar.close()
            log.info("\n" + "=" * 60)
            log.info("RÉSUMÉ DES MEILLEURS HYPERPARAMÈTRES")
            log.info("=" * 60)
            for mt, params in best_params_all.items():
                log.info(f"\n[{mt}]")
                for k, v in params.items():
                    log.info(f"  {k:30s} = {v}")
            log.info(f"\nDurée totale : {time.time() - t_global:.0f}s")
            return

        for mt in models_to_run:
            step(f"Entraînement {mt}")
            model, metrics = train_final(
                mt, best_params_all[mt],
                X_train, y_train, X_eval, y_eval,
            )
            save_artifacts(mt, model, metrics, feature_names, output_dir)
            all_metrics[mt] = metrics
            pipeline_bar.update(1)

    # --- 5. Mode TRAIN ---
    elif args.mode == "train":
        log.info(f"MODE TRAIN — {', '.join(models_to_run)}")

        for mt in models_to_run:
            step(f"Entraînement {mt}")
            params_path = output_dir / f"{mt}_best_params.json"
            if params_path.exists():
                params = json.loads(params_path.read_text())
                log.info(f"[{mt}] Paramètres chargés depuis {params_path}")
            else:
                params = DEFAULT_PARAMS[mt]
                log.info(f"[{mt}] Paramètres par défaut utilisés")

            model, metrics = train_final(mt, params, X_train, y_train, X_eval, y_eval)
            save_artifacts(mt, model, metrics, feature_names, output_dir)
            all_metrics[mt] = metrics
            pipeline_bar.update(1)

    pipeline_bar.close()

    # --- 6. Résumé final ---
    if all_metrics:
        elapsed = time.time() - t_global
        log.info("\n" + "=" * 60)
        log.info("RÉSUMÉ FINAL")
        log.info(f"{'Modèle':<20} {'CV MAE':>10} {'±':>8} {'Eval MAE':>10}")
        log.info("-" * 50)
        for mt, m in all_metrics.items():
            log.info(
                f"{mt:<20} {m['cv_mae']:>10.4f} {m['cv_std']:>8.4f} {m['test_mae']:>10.4f}"
            )
        log.info(f"\nDurée totale : {elapsed:.0f}s")
        log.info("=" * 60)

        end_path = "summary_" + args.model + ".json"
        summary_path = output_dir / end_path
        summary_path.write_text(json.dumps(all_metrics, indent=2))
        log.info(f"Résumé complet → {summary_path}")


# ---------------------------------------------------------------------------
# Entrée CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tuning et entraînement de modèles RF / LightGBM / XGBoost "
                    "pour la prévision day-ahead de production éolienne offshore."
    )
    parser.add_argument(
        "--data_folder", type=str, default="data/",
        help="Dossier contenant les fichiers .parquet",
    )
    parser.add_argument(
        "--mode", choices=["tune", "train", "all"], default="all",
        help="tune=Optuna seul | train=entraîne avec params existants | all=tune+train",
    )
    parser.add_argument(
        "--model",
        choices=["random_forest", "lightgbm", "xgboost", "all"],
        default="all",
        help="Modèle(s) ciblé(s), s'applique à tous les modes (défaut : all)",
    )
    parser.add_argument(
        "--train_percent", type=float, default=0.8,
        help="Pourcentage du dataset à inclure dans l'ensemble de train (80% par défaut)",
    )
    parser.add_argument(
        "--n_trials", type=int, default=50,
        help="Nombre d'essais Optuna par modèle (défaut : 50)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/",
        help="Dossier de sortie pour les modèles et métriques",
    )
    parser.add_argument(
        "--site", type=str, default="all",
        help="Site sur lequel entraîner le modèle (si all, crée un modèle général)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.site == "all_indiv":
        for site in SITE_LIST:
            site_args = argparse.Namespace(data_folder=args.data_folder,
                                      mode=args.mode,
                                      model=args.model,
                                      train_percent=args.train_percent,
                                      n_trials=args.n_trials,
                                      output_dir=os.path.join(args.output_dir, site),
                                      site= site,
                                      )
            run_pipeline(site_args)