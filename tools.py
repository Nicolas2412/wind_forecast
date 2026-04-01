from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

DEFAULT_TEST_SIZE = 0.2

SUPPORTED_MODELS = ["random_forest", "xgboost", "lightgbm", "sarimax", "lstm", "transformer"]

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
    },
    "lightgbm": {
        "n_estimators": 200,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "sarimax": {
        "order": [1, 1, 1],
        "seasonal_order": [1, 1, 1, 24],
        "enforce_stationarity": False,
        "enforce_invertibility": False,
    },
    "lstm": {
        "seq_len": 48,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "epochs": 30,
        "batch_size": 256,
        "learning_rate": 1e-3,
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
    },
}


def _load_config(config_path: str = "config.yaml") -> dict:
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path(__file__).resolve().parent / config_file
    if not config_file.exists():
        return {
            "valid_models": SUPPORTED_MODELS,
            "model_params": deepcopy(DEFAULT_MODEL_PARAMS),
        }

    try:
        import yaml
    except ImportError:
        return {
            "valid_models": SUPPORTED_MODELS,
            "model_params": deepcopy(DEFAULT_MODEL_PARAMS),
        }

    try:
        with config_file.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
    except Exception:
        raw = {}

    config_valid_models = raw.get("valid_models", SUPPORTED_MODELS)
    valid_models = [
        model_name
        for model_name in config_valid_models
        if model_name in SUPPORTED_MODELS
    ]
    if not valid_models:
        valid_models = list(SUPPORTED_MODELS)

    model_params = deepcopy(DEFAULT_MODEL_PARAMS)
    loaded_model_params = raw.get("model_params", {})
    if isinstance(loaded_model_params, dict):
        for model_name, params in loaded_model_params.items():
            if model_name in model_params and isinstance(params, dict):
                model_params[model_name].update(params)

    return {
        "valid_models": valid_models,
        "model_params": model_params,
    }


_APP_CONFIG = _load_config()
VALID_MODELS = _APP_CONFIG["valid_models"]
MODEL_PARAMS = _APP_CONFIG["model_params"]
DEFAULT_MODEL_TYPE = VALID_MODELS[0]

# ---------------------------------------------------------------------------
# PyTorch helpers (imported lazily inside the classes that need them)
# ---------------------------------------------------------------------------


def _build_lstm_net(input_size: int, hidden_size: int, num_layers: int, dropout: float):
    """Build a PyTorch LSTM regressor with custom weight initialization."""
    import torch
    import torch.nn as nn

    class _LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out).squeeze(-1)

    # --- INITIALISATION DES POIDS ---
    def init_weights(m):
        if isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    # Xavier pour les entrées vers hidden
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    # Orthogonal pour le récurrent (maintient la norme du gradient)
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    # On met les biais à 0, sauf parfois le biais de la porte d'oubli
                    param.data.fill_(0)
        elif isinstance(m, nn.Linear):
            # Xavier pour la couche de sortie
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)

    model = _LSTMNet()
    model.apply(init_weights)  # Applique l'initialisation à tout le réseau
    return model


def _build_transformer_net(input_size: int, d_model: int, nhead: int,
                            num_layers: int, dropout: float):
    """Build a PyTorch Transformer regressor."""
    import torch
    import torch.nn as nn

    class _TransformerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout,
                dim_feedforward=d_model * 4, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):                       # x: (B, T, F)
            x = self.input_proj(x)                  # (B, T, d_model)
            x = self.encoder(x)                     # (B, T, d_model)
            x = x[:, -1, :]                         # last timestep (B, d_model)
            return self.fc(x).squeeze(-1)            # (B,)

    return _TransformerNet()

# ---------------------------------------------------------------------------
# DataProcessor
# ---------------------------------------------------------------------------

class DataProcessor:
    """Data Processing class."""

    def __init__(self, path_folder: str, X: pd.DataFrame = None, drop_columns: list = []):
        self.path = path_folder
        self.time_column = "delivery_time"
        self.predict_column = "production_normalized"
        self.drop_columns = list(set(
            drop_columns + ['production', 'installed_capacity', 'is_not_plateau']))
        self.df = self.open_data() if X is None else X
        
    def run(self) -> pd.DataFrame:
        """Run the full processing pipeline."""
        df = self.prepocess_data()
        df = self.impute_production(df)
        df = self.engineer_features(df)
        return df

    def open_data(self) -> pd.DataFrame:
        """Open and merge data while fixing type mismatches"""
        main_df = None
        files = sorted(Path(self.path).glob("*.parquet"))
        
        for file in files:
            if "dataset_2" in file.name:
                continue
                
            current_df = pd.read_parquet(file)
            if "delivery_time" in current_df.columns:
                current_df["delivery_time"] = pd.to_datetime(current_df["delivery_time"], utc=True)
            if main_df is None:
                main_df = current_df
            else:
                main_df["delivery_time"] = pd.to_datetime(main_df["delivery_time"], utc=True)
                main_df = pd.merge(
                    main_df, 
                    current_df, 
                    on=["site_name", "delivery_time"], 
                    how="inner"
                )
        return main_df if main_df is not None else pd.DataFrame()

    def prepocess_data(self,
                    N: int = 5,
                    window: str = "24h",
                    tolerance: float = 0.01,
                    low_thresh: float = 0.1,
                    high_thresh: float = 0.9) -> pd.DataFrame:

        df = self.df.copy()

        df["production_normalized"] = df["production"] / df["installed_capacity"]

        df = compute_plateau(df=df, N=N, window=window, tolerance=tolerance, low_thresh=low_thresh, high_thresh=high_thresh)
        
        self.df = df
        return df

    def impute_production(self, df: pd.DataFrame, max_gap_hours: int = 6) -> pd.DataFrame:
        results = []
        for site, grp in df.groupby("site_name"):
            # On s'assure que le temps est bien au format datetime
            grp[self.time_column] = pd.to_datetime(grp[self.time_column])
            grp = grp.sort_values(self.time_column).copy()

            # 1. Plateaux → NaN
            if "is_not_plateau" in grp.columns:
                grp.loc[~grp["is_not_plateau"], "production_normalized"] = np.nan

            # --- FIX ICI ---
            # On passe la colonne temporelle en index pour permettre l'interpolation "time"
            grp = grp.set_index(self.time_column)

            # 2. Interpolation temporelle
            grp["production_normalized"] = (
                grp["production_normalized"]
                .interpolate(method="time", limit=max_gap_hours)
            )

            # On repasse en index numérique pour la suite des opérations (fillna, etc.)
            grp = grp.reset_index()
            # ---------------

            # 3. Trous longs résiduels → médiane du site
            site_median = grp["production_normalized"].median()
            grp["production_normalized"] = grp["production_normalized"].fillna(
                site_median)

            # 4. ffill/bfill pour les bords
            grp["production_normalized"] = (
                grp["production_normalized"].ffill().bfill()
            )

            results.append(grp)

        return pd.concat(results, ignore_index=True)
    
    
    def engineer_features(self, df: pd.DataFrame, data_delay_days: int = 15) -> pd.DataFrame:
        """Engineer features for the model."""
        df = df.copy()
        df = df.dropna(subset=[self.time_column, "production", "installed_capacity"])
        df = df[df["installed_capacity"] != 0]

        df[self.time_column] = pd.to_datetime(df[self.time_column])
        df["hour"]        = df[self.time_column].dt.hour
        df["day_of_week"] = df[self.time_column].dt.dayofweek
        df["month"]       = df[self.time_column].dt.month
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
        df["is_night"]    = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
        df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Evite les gros pics
        df['precipitation'] = np.log1p(df['precipitation'])
        
        df["production_normalized"] = df["production"] / df["installed_capacity"]

        df["wind_speed_diff"]  = df["wind_speed_100m"] - df["wind_speed_10m"]
        for col in ["wind_speed_10m", "wind_speed_100m"]:
            df[f"{col}_squared"] = df[col] ** 2
            df[f"{col}_cubed"]   = df[col] ** 3
            
        df["wind_speed_ratio"] = df["wind_speed_100m"] / (df["wind_speed_10m"] + 1e-8)
        v10 = df["wind_speed_10m"].clip(lower=0.5)
        v100 = df["wind_speed_100m"].clip(lower=0.5)
        df["wind_shear_alpha"] = np.log(v100 / v10) / np.log(100 / 10)
        
        min_lag_h = data_delay_days * 24   # 360h pour 15 jours
        
        results = []
        for site, grp in df.groupby("site_name"):
            grp = grp.set_index(self.time_column).sort_index()

            # --- Grille horaire complète (corrige les trous de maintenance) ---
            # Pas sur que ca soit utile comme on le fait avant de retirer les plateaux
            full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq="1h", tz="UTC")
            grp = grp.reindex(full_idx)
            grp["site_name"] = site

            # --- Lags causaux (>= data_delay_days * 24h) ---
            # lag_360h : même heure, J-15 — premier lag disponible
            grp[f"production_lag{min_lag_h}h"] = grp["production_normalized"].shift(
                min_lag_h)
            # lag_384h : même heure, J-16
            grp[f"production_lag{min_lag_h + 24}h"] = grp["production_normalized"].shift(
                min_lag_h + 24)
            # lag_720h : même heure, J-30 (capture la saisonnalité mensuelle)
            grp["production_lag720h"] = grp["production_normalized"].shift(720)

            # --- Statistiques glissantes causales ---
            # On shifte d'abord de min_lag_h pour que la fenêtre commence à J-15
            shifted = grp["production_normalized"].shift(min_lag_h)
            # Fenêtre 7 jours (168h) sur [J-15, J-22]
            grp["production_rolling_mean_7d"] = shifted.rolling(
                168, min_periods=84).mean()
            grp["production_rolling_std_7d"] = shifted.rolling(
                168, min_periods=84).std()
            # Fenêtre 14 jours (336h) sur [J-15, J-29]
            grp["production_rolling_mean_14d"] = shifted.rolling(
                336, min_periods=168).mean()
            grp["production_rolling_std_14d"] = shifted.rolling(
                336, min_periods=168).std()

            # --- Suppression des lignes de maintenance APRÈS calcul des lags ---
            if "is_not_plateau" in grp.columns:
                grp = grp[grp["is_not_plateau"].fillna(False).infer_objects(copy=False).astype(bool)]

            results.append(grp.reset_index().rename(columns={"index": self.time_column}))

        df = pd.concat(results, ignore_index=True)
        
        # --- Features physiques vent (issues du NWP, toujours disponibles) ---
        if "wind_direction_100m" in df.columns:
            rad = np.radians(df["wind_direction_100m"])
            # Convention météo : direction = origine du vent (0°=Nord, 90°=Est)
            # U (zonal, positif vers l'Est)     = -speed * sin(dir)
            # V (méridional, positif vers le Nord) = -speed * cos(dir)
            df["wind_u_100m"] = -df["wind_speed_100m"] * np.sin(rad)
            df["wind_v_100m"] = -df["wind_speed_100m"] * np.cos(rad)
        
        if "wind_direction_10m" in df.columns:
            rad = np.radians(df["wind_direction_10m"])
            # Convention météo : direction = origine du vent (0°=Nord, 90°=Est)
            # U (zonal, positif vers l'Est)     = -speed * sin(dir)
            # V (méridional, positif vers le Nord) = -speed * cos(dir)
            df["wind_u_10m"] = -df["wind_speed_10m"] * np.sin(rad)
            df["wind_v_10m"] = -df["wind_speed_10m"] * np.cos(rad)

        if "temperature_2m" in df.columns and "pressure_msl" in df.columns:
            # Densité de l'air : ρ = P / (R_d * T),  R_d = 287.05 J/(kg·K)
            df["air_density"] = df["pressure_msl"] / \
                (287.05 * (df["temperature_2m"] + 273.15))
            # Puissance éolienne théorique ∝ ρ·v³ (avant coefficient de puissance Cp)
            df["theoretical_power"] = df["air_density"] * df["wind_speed_100m"] ** 3
        
        df = df.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)

        return df
    
    def finalize_for_model(self, df):
        # 1. On définit ce qu'on veut exclure
        # Note: J'ajoute "site_name" ici si tu ne le veux pas dans le modèle final
        to_exclude = self.drop_columns

        # 2. On identifie les colonnes à GARDER (features + cible)
        # On s'assure de ne garder que ce qui existe réellement dans le df
        cols_to_keep = [c for c in df.columns if c not in to_exclude]

        # 3. Création du masque de lignes (on vérifie les NaN sur les colonnes conservées)
        # On vérifie que TOUTES les colonnes gardées sont non-nulles sur la ligne
        mask = df[cols_to_keep].notna().all(axis=1)

        # 4. On filtre les LIGNES avec le masque ET les COLONNES avec cols_to_keep
        df_final = df.loc[mask, cols_to_keep]

        return df_final

def compute_plateau(df: pd.DataFrame, N: int = 5, window: str = "24h", tolerance: float = 0.01, low_thresh: float = 0.1, high_thresh: float = 0.9):

    def count_similar_in_window(series, window, tol):
        # Déduire le pas temporel
        freq = series.index.to_series().diff().median()
        half_window = pd.Timedelta(window) / 2
        n_points = int(half_window / freq)  # nb de points de chaque côté

        values = series.values
        counts = np.array([
            np.sum(np.abs(values[max(0, i-n_points):i+n_points+1] - v) < tol)
            for i, v in enumerate(values)
        ])
        return pd.Series(counts, index=series.index)

    results = []
    for site, df_site in df.groupby("site_name"):
        df_site = df_site.sort_values("delivery_time").copy()
        df_site = df_site.set_index("delivery_time")

        p_max = df_site["production_normalized"].max()
        p_low = p_max * low_thresh
        p_high = p_max * high_thresh

        df_site["similar_count"] = count_similar_in_window(
            df_site["production_normalized"], window, tolerance)
        in_zone = (df_site["production_normalized"] >= p_low) & (
            df_site["production_normalized"] <= p_high)
        df_site["is_not_plateau"] = ~(
            (df_site["similar_count"] >= N) & in_zone)

        results.append(df_site.drop(columns=["similar_count"]).reset_index())
        
    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# ForecastModel
# ---------------------------------------------------------------------------

class ForecastModel:
    """
    Unified forecasting model supporting:
        random_forest | xgboost | lightgbm | sarimax | lstm | transformer
    """

    def __init__(self, model_type: str = DEFAULT_MODEL_TYPE, savepath:str = None):
        self.time_column    = "delivery_time"
        self.predict_column = "production_normalized"
        self.n_splits       = 5
        self.model_type     = self._validate(model_type)
        self.SKLEARN_DEFAULTS = dict(n_estimators=200, random_state=42, n_jobs=-1)
        self.XGBOOST_TREE_METHOD = "hist"
        self.LIGHTGBM_VERBOSE = -1
        self.SARIMAX_ORDER = (1, 1, 1)
        self.SARIMAX_SEAS_ORDER = (1, 1, 1, 24)
        self.SARIMAX_ENFORCE_STATIONARITY = False
        self.SARIMAX_ENFORCE_INVERTIBILITY = False
        self.LSTM_SEQ_LEN = 48
        self.LSTM_HIDDEN = 128
        self.LSTM_LAYERS = 2
        self.LSTM_DROPOUT = 0.2
        self.TRANSFORMER_SEQ_LEN = 48
        self.TRANSFORMER_D_MODEL = 64
        self.TRANSFORMER_NHEAD = 4
        self.TRANSFORMER_LAYERS = 2
        self.TRANSFORMER_DROPOUT = 0.1
        self.DL_EPOCHS = 30
        self.DL_BATCH_SIZE = 256
        self.DL_LR = 5e-5
        self.DL_GRAD_CLIP = 1.0
        # self._apply_model_params_from_config()
        self.model          = None          # built lazily or at train time
        self.scaler_X       = None          # used by LSTM / Transformer
        self.scaler_y       = None          # used by LSTM / Transformer
        self.evaluation_results: dict = {}
        self.savepath = savepath
        
        
        self.evaluation_results: dict = {}
        self.savepath = savepath
        
        if self.savepath:
            p = Path(self.savepath)
            # Dossier basé sur le nom du fichier (ex: models/lstm_v1_results/)
            self.output_dir = p.parent / f"{p.stem}_results"
            self.eval_dir = self.output_dir / "evaluation"
            
            # On crée les dossiers s'ils n'existent pas
            self.eval_dir.mkdir(parents=True, exist_ok=True)
            
        #  Vérification si un modèle existe déjà au chemin spécifié
        if self.savepath and Path(self.savepath).exists():
            print(f"[{model_type}] Un modèle existant a été trouvé à {self.savepath}. Chargement...")
            try:
                self.load(self.savepath)
            except Exception as e:
                print(f"Erreur lors du chargement automatique : {e}")
                print("Le modèle sera réinitialisé.")
                self.model = None
        else:
            # 2. Si pas de fichier existant, on initialise normalement
            self.model = None 
            self.scaler_X = StandardScaler()
            self.scaler_y = MinMaxScaler(feature_range=(0, 1))
            
            # Initialisation légère pour les modèles tree-based
            if self.model_type in ("random_forest", "xgboost", "lightgbm"):
                self.model = self._build_sklearn_model()
                
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> None:
        """Train with time-series cross-validation, then refit on full data."""
        dispatch = {
            "random_forest": self._train_sklearn,
            "xgboost":       self._train_sklearn,
            "lightgbm":      self._train_sklearn,
            "sarimax":       self._train_sarimax,
            "lstm":          self._train_deep,
            "transformer":   self._train_deep,
        }
        dispatch[self.model_type](df)

    def evaluate(self, df: pd.DataFrame = None, plot: bool = False) -> dict:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np

        results = dict(self.evaluation_results)
        results["per_site_metrics"] = {}
        if df is not None:
            if self.model_type in ['lstm', 'transformer']:
                prediction, y_true = self._predict_deep(df)
                seq_len = self.LSTM_SEQ_LEN
            else:
                prediction = self.predict(df)
                y_true = df[self.predict_column].to_numpy()
                seq_len = 0

            # Calcul des métriques
            mae = mean_absolute_error(y_true, prediction)
            mse = mean_squared_error(y_true, prediction)
            rmse = np.sqrt(mse)

            # nRMSE (normalisée par la moyenne pour donner un %)
            mean_y = np.mean(y_true)
            nrmse = (rmse / mean_y) if mean_y != 0 else 0

            results.update({
                "eval_mae": float(mae),
                "eval_rmse": float(rmse),
                "eval_nrmse": float(nrmse)
            })
            
            df[self.time_column] = pd.to_datetime(
                df[self.time_column]).dt.tz_localize(None)
            all_dates = sorted(df[self.time_column].unique())
            sum_df = pd.DataFrame(index=all_dates)
            sum_df['total_true'] = 0.0
            sum_df['total_pred'] = 0.0
            sum_df['count'] = 0
            

            site_names = df['site_name'].unique()
            current_idx = 0
            
            n_cols = 2
            n_rows = (len(site_names) // n_cols) + 1
            fig = plt.figure(figsize=(16, n_rows * 4.5))
            fig.suptitle(f"Évaluation Multi-Sites (300 dernières heures): Modèle {self.model_type.upper()}\nMAE Global: {mae:.4f} | RMSE Global: {rmse:.4f}",
                        fontsize=18, fontweight='bold', y=0.98)
            for i, site in enumerate(site_names):
                # Calcul de la longueur attendue pour ce site
                # (Nombre de lignes du site - seq_len)
                site_data = df[df['site_name'] == site]
                site_data_len = len(df[df['site_name'] == site])
                n_seq = site_data_len - seq_len

                if n_seq <= 0:
                    continue

                # Extraction des segments
                site_pred = prediction[current_idx: current_idx + n_seq]
                site_true = y_true[current_idx: current_idx + n_seq]
                
                site_dates = pd.to_datetime(
                    site_data[self.time_column].iloc[seq_len:].values)

                # 2. On crée un DataFrame temporaire pour ce site
                temp_site = pd.DataFrame({
                    'total_true': site_true.flatten(),
                    'total_pred': site_pred.flatten(),
                    'count': 1
                }, index=site_dates)

                # 3. On ajoute au global en utilisant l'index pour aligner
                # On utilise add() avec fill_value=0 pour gérer les dates manquantes
                sum_df = sum_df.add(temp_site, fill_value=0)

                s_mae = float(mean_absolute_error(site_true, site_pred))
                s_rmse = float(np.sqrt(mean_squared_error(site_true, site_pred)))
                
                results["per_site_metrics"][site] = {
                    "mae": s_mae,
                    "rmse": s_rmse
                }
                
                
                # Plot individuel
                ax = plt.subplot(n_rows, n_cols, i + 1)
                ax.plot(site_true[-300:], color='blue',
                        alpha=0.7, label='Réel' if i == 0 else "")
                ax.plot(site_pred[-300:], color='red', linestyle='--',
                        alpha=0.8, label='Prédit' if i == 0 else "")

                ax.set_title(f"Site: {site} - MAE: {mean_absolute_error(site_true, site_pred):.4f}",
                            fontsize=12, pad=10)

                # On n'affiche la légende que pour le premier plot pour ne pas répéter
                if i == 0:
                    ax.legend(loc='upper left',fontsize=12)
                ax.grid(True, alpha=0.2)
                current_idx += n_seq
                
            plt.subplots_adjust(
                left=0.08,
                right=0.92,
                top=0.94,
                bottom=0.06,
                hspace=0.45,  # L'espace "respirable" mais pas vide
                wspace=0.22
            )
            img_name = f"eval_plots_{self.model_type}.png"

            # Chemin final : soit dans le dossier dédié, soit à la racine du projet
            save_path = self.eval_dir / img_name if self.eval_dir else img_name

            plt.savefig(save_path, bbox_inches='tight', dpi=120)
            print(f"\n> Graphique Multi-sites sauvegardé sous : {save_path}")
            
            # On ne garde que les moments où on a des données (count > 0)
            # On prend les 300 derniers points communs
            valid_sum = sum_df[sum_df['count'] > 0].tail(300)

            plt.figure(figsize=(15, 7))
            plt.plot(valid_sum.index, valid_sum['total_true'],
                    color='navy', label='Production Totale Réelle', linewidth=2)
            plt.plot(valid_sum.index, valid_sum['total_pred'], color='crimson',
                    linestyle='--', label='Production Totale Prédite', linewidth=2)

            sum_mae = mean_absolute_error(
                valid_sum['total_true'], valid_sum['total_pred'])
            n_sites = len(site_names)
            mae_par_site_equivalent = sum_mae / n_sites
            plt.title(
                f"Somme de tout les sites({len(site_names)} sites)\nMAE Sommé: {sum_mae:.4f} (Moyenne par site: {mae_par_site_equivalent:.4f})", fontsize=15, fontweight='bold')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            sum_img_name = f"eval_total_sum_{self.model_type}.png"
            sum_save_path = self.eval_dir / sum_img_name if self.eval_dir else sum_img_name

            plt.savefig(sum_save_path, bbox_inches='tight', dpi=120)
            print(f"> Graphique de la somme sauvegardé sous : {sum_save_path}")
            
            n_sites = len(site_names)
            full_portfolio_data = sum_df[sum_df['count'] == n_sites]

            portfolio_mae_total = 0.0
            portfolio_mae_per_site = 0.0
            portfolio_rmse_total = 0.0
            portfolio_rmse_per_site = 0.0

            if len(full_portfolio_data) > 0:
                # MAE
                portfolio_mae_total = float(mean_absolute_error(
                    full_portfolio_data['total_true'],
                    full_portfolio_data['total_pred']
                ))
                portfolio_mae_per_site = portfolio_mae_total / n_sites

                # RMSE (La partie manquante)
                portfolio_rmse_total = float(np.sqrt(mean_squared_error(
                    full_portfolio_data['total_true'],
                    full_portfolio_data['total_pred']
                )))
                portfolio_rmse_per_site = portfolio_rmse_total / n_sites

            # Mise à jour du dictionnaire avec TOUTES les métriques
            results.update({
                "portfolio_mae_total": portfolio_mae_total,
                "portfolio_mae_per_site": portfolio_mae_per_site,
                "portfolio_rmse_total": portfolio_rmse_total,
                "portfolio_rmse_per_site": portfolio_rmse_per_site,
                "n_sites": n_sites
            })
            
            
            if plot:
                plt.show()
                
            

        return results
    

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict on a new DataFrame."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        dispatch = {
            "random_forest": self._predict_sklearn,
            "xgboost":       self._predict_sklearn,
            "lightgbm":      self._predict_sklearn,
            "sarimax":       self._predict_sarimax,
            "lstm":          self._predict_deep,
            "transformer":   self._predict_deep,
        }
        return dispatch[self.model_type](df)

    def save(self, path: str = None) -> None:
        """Save the trained model to disk."""
        import joblib
        save_path = path or self.savepath
        if not save_path:
            raise ValueError("No save path provided.")
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        save_path = str(save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        if self.model_type in ("lstm", "transformer"):
            import torch
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'input_size': getattr(self, 'input_size', None)
            }
            torch.save(checkpoint, save_path)
        elif self.model_type == "sarimax":
            self.model.save(save_path)
        else:
            joblib.dump(self.model, save_path)
        print(f"[{self.model_type}] Model saved to {save_path}")

    def load(self, path: str = None) -> None:
        """Load a trained model from disk."""
        import joblib
        load_path = path or self.savepath
        if not load_path:
            raise ValueError("No load path provided.")
            
        load_path = str(load_path)
        if not Path(load_path).exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
            
        if self.model_type in ("lstm", "transformer"):
            import torch
            checkpoint = torch.load( load_path, map_location=torch.device('cpu'), weights_only=False)
            self.input_size = checkpoint['input_size']
            self.scaler_X = checkpoint['scaler_X']
            self.scaler_y = checkpoint['scaler_y']
            self.model = self._build_dl_model(self.input_size)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif self.model_type == "sarimax":
            import statsmodels.api as sm
            self.model = sm.load(load_path)
        else:
            self.model = joblib.load(load_path)
        print(f"[{self.model_type}] Model loaded from {load_path}")

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------

    def _validate(self, model_type: str) -> str:
        self.VALID_MODELS = VALID_MODELS
        if model_type not in self.VALID_MODELS:
            raise ValueError(f"Invalid model type. Choose from {self.VALID_MODELS}")
        return model_type

    def _build_sklearn_model(self):
        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**self.SKLEARN_DEFAULTS)
        if self.model_type == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=self.SKLEARN_DEFAULTS["n_estimators"],
                random_state=self.SKLEARN_DEFAULTS["random_state"],
                n_jobs=self.SKLEARN_DEFAULTS["n_jobs"],
                tree_method=self.XGBOOST_TREE_METHOD,
            )
        if self.model_type == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=self.SKLEARN_DEFAULTS["n_estimators"],
                random_state=self.SKLEARN_DEFAULTS["random_state"],
                n_jobs=self.SKLEARN_DEFAULTS["n_jobs"],
                verbose=self.LIGHTGBM_VERBOSE,
            )

    def _apply_model_params_from_config(self) -> None:
        params = MODEL_PARAMS.get(self.model_type, {})

        if self.model_type in ("random_forest", "xgboost", "lightgbm"):
            for key in ("n_estimators", "random_state", "n_jobs"):
                if key in params:
                    self.SKLEARN_DEFAULTS[key] = params[key]
            if self.model_type == "xgboost" and "tree_method" in params:
                self.XGBOOST_TREE_METHOD = params["tree_method"]
            if self.model_type == "lightgbm" and "verbose" in params:
                self.LIGHTGBM_VERBOSE = params["verbose"]
            return

        if self.model_type == "sarimax":
            if "order" in params:
                self.SARIMAX_ORDER = tuple(params["order"])
            if "seasonal_order" in params:
                self.SARIMAX_SEAS_ORDER = tuple(params["seasonal_order"])
            if "enforce_stationarity" in params:
                self.SARIMAX_ENFORCE_STATIONARITY = params["enforce_stationarity"]
            if "enforce_invertibility" in params:
                self.SARIMAX_ENFORCE_INVERTIBILITY = params["enforce_invertibility"]
            return

        if self.model_type == "lstm":
            self.LSTM_SEQ_LEN = params.get("seq_len", self.LSTM_SEQ_LEN)
            self.LSTM_HIDDEN = params.get("hidden_size", self.LSTM_HIDDEN)
            self.LSTM_LAYERS = params.get("num_layers", self.LSTM_LAYERS)
            self.LSTM_DROPOUT = params.get("dropout", self.LSTM_DROPOUT)
            self.DL_EPOCHS = params.get("epochs", self.DL_EPOCHS)
            self.DL_BATCH_SIZE = params.get("batch_size", self.DL_BATCH_SIZE)
            self.DL_LR = params.get("learning_rate", self.DL_LR)
            return

        if self.model_type == "transformer":
            self.TRANSFORMER_SEQ_LEN = params.get("seq_len", self.TRANSFORMER_SEQ_LEN)
            self.TRANSFORMER_D_MODEL = params.get("d_model", self.TRANSFORMER_D_MODEL)
            self.TRANSFORMER_NHEAD = params.get("nhead", self.TRANSFORMER_NHEAD)
            self.TRANSFORMER_LAYERS = params.get("num_layers", self.TRANSFORMER_LAYERS)
            self.TRANSFORMER_DROPOUT = params.get("dropout", self.TRANSFORMER_DROPOUT)
            self.DL_EPOCHS = params.get("epochs", self.DL_EPOCHS)
            self.DL_BATCH_SIZE = params.get("batch_size", self.DL_BATCH_SIZE)
            self.DL_LR = params.get("learning_rate", self.DL_LR)

    # ------------------------------------------------------------------
    # sklearn / tree training  (fixes the variable-shadowing bug)
    # ------------------------------------------------------------------

    def _train_sklearn(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=[self.time_column, self.predict_column])
        y = df[self.predict_column]

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        maes = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            fold_model = clone(self.model)
            fold_model.fit(X_tr, y_tr)
            maes.append(mean_absolute_error(y_val, fold_model.predict(X_val)))

        cv_mae = float(np.mean(maes))
        self.evaluation_results = {"cv_mae": cv_mae, "cv_mae_std": float(np.std(maes))}
        print(f"[{self.model_type}] CV MAE: {cv_mae:.4f} ± {np.std(maes):.4f}")

        self.model.fit(X, y)

    def _predict_sklearn(self, df: pd.DataFrame) -> np.ndarray:
        X = df.drop(columns=[self.time_column, self.predict_column], errors="ignore")
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # SARIMAX
    # ------------------------------------------------------------------

    def _train_sarimax(self, df: pd.DataFrame) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        series = (
            df.set_index(self.time_column)[self.predict_column]
            .sort_index()
        )
        if series.index.has_duplicates:
            series = series.groupby(level=0).mean()
        series = series.asfreq("h").interpolate(method="time").ffill().bfill()

        tscv  = TimeSeriesSplit(n_splits=self.n_splits)
        idx   = np.arange(len(series))
        maes  = []

        for train_idx, val_idx in tscv.split(idx):
            train_s = series.iloc[train_idx]
            val_s   = series.iloc[val_idx]
            try:
                res = SARIMAX(
                    train_s,
                    order=self.SARIMAX_ORDER,
                    seasonal_order=self.SARIMAX_SEAS_ORDER,
                    enforce_stationarity=self.SARIMAX_ENFORCE_STATIONARITY,
                    enforce_invertibility=self.SARIMAX_ENFORCE_INVERTIBILITY,
                ).fit(disp=False)
                preds = res.forecast(steps=len(val_s))
                maes.append(mean_absolute_error(val_s, preds))
            except Exception as exc:
                print(f"  SARIMAX fold skipped: {exc}")

        cv_mae = float(np.mean(maes)) if maes else float("nan")
        self.evaluation_results = {"cv_mae": cv_mae}
        print(f"[sarimax] CV MAE: {cv_mae:.4f}")

        self.model = SARIMAX(
            series,
            order=self.SARIMAX_ORDER,
            seasonal_order=self.SARIMAX_SEAS_ORDER,
            enforce_stationarity=self.SARIMAX_ENFORCE_STATIONARITY,
            enforce_invertibility=self.SARIMAX_ENFORCE_INVERTIBILITY,
        ).fit(disp=False)

    def _predict_sarimax(self, df: pd.DataFrame) -> np.ndarray:
        n_steps = len(df)
        forecast = self.model.forecast(steps=n_steps)
        return forecast.values

    # ------------------------------------------------------------------
    # Deep-learning helpers (LSTM & Transformer share the same loop)
    # ------------------------------------------------------------------

    def _to_arrays(self, frame, feat_cols, target_col):
        # Juste pour transformer un DF en matrice numpy propre pour les scalers
        mask = frame[feat_cols].notna().all(
            axis=1) & frame[target_col].notna()
        f = frame[mask]
        return f[feat_cols].values.astype(np.float32), f[target_col].values.astype(np.float32), mask

    def _get_dl_params(self) -> dict:
        if self.model_type == "lstm":
            return dict(
                seq_len=self.LSTM_SEQ_LEN,
                epochs=self.DL_EPOCHS,
                batch=self.DL_BATCH_SIZE,
            )
        return dict(
            seq_len=self.TRANSFORMER_SEQ_LEN,
            epochs=self.DL_EPOCHS,
            batch=self.DL_BATCH_SIZE,
        )

    def _build_dl_model(self, input_size: int):
        if self.model_type == "lstm":
            return _build_lstm_net(
                input_size=input_size,
                hidden_size=self.LSTM_HIDDEN,
                num_layers=self.LSTM_LAYERS,
                dropout=self.LSTM_DROPOUT,
            )
        return _build_transformer_net(
            input_size=input_size,
            d_model=self.TRANSFORMER_D_MODEL,
            nhead=self.TRANSFORMER_NHEAD,
            num_layers=self.TRANSFORMER_LAYERS,
            dropout=self.TRANSFORMER_DROPOUT,
        )
        
    def _prepare_sequences(self, df, feat_cols, scaler_X, scaler_y, seq_len):
        """
        Scaling par site + Fenêtrage NumPy + Recollage global.
        """
        all_X, all_y = [], []
        
        # On boucle par site pour ne JAMAIS mélanger la fin d'un site avec le début d'un autre
        for site, grp in df.groupby("site_name"):
            if len(grp) <= seq_len:
                continue

            # 1. Extraction des valeurs brutes pour ce site
            X_site_raw = grp[feat_cols].values.astype(np.float32)
            y_site_raw = grp[self.predict_column].values.astype(
                np.float32).reshape(-1, 1)

            # 2. Application du scaling (fit à l'extérieur, transform ici)
            X_site_sc = scaler_X.transform(X_site_raw)
            X_site_sc = np.clip(X_site_sc, -5.0, 5.0) #Pour etre sur de pas avoir de valeur aberantes
            y_site_sc = scaler_y.transform(y_site_raw).ravel()

            # 3. Fenêtrage glissante
            for i in range(len(X_site_sc) - seq_len):
                all_X.append(X_site_sc[i: i + seq_len])
                all_y.append(y_site_sc[i + seq_len])

        if not all_X:
            return np.array([]), np.array([])

        return np.array(all_X), np.array(all_y)

    def _train_deep(self, df: pd.DataFrame):
        import copy
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        feat_cols = [c for c in df.columns if c not in (
            self.time_column,
            self.predict_column,
            'site_name'
        )]
        print(f"Feat cols:  {feat_cols}")

        params = self._get_dl_params()
        seq_len = params["seq_len"]
        # Need to keep unique times in distinct splits (each site has a row for each time)
        unique_times = np.sort(df[self.time_column].unique())
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        # ------------------------------------------------------------------ #
        #  Phase 1 : Cross-validation                                         #
        #  But : estimer la perf + calibrer le nombre d'epochs pour le refit  #
        # ------------------------------------------------------------------ #
        fold_maes, fold_best_epochs = [], []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(unique_times)):
            print(f"\n--- Cross validation Fold {fold+1}/{self.n_splits} ---")

            # --- A. DÉFINITION DES PÉRIODES (DATES) ---
            train_dates = unique_times[train_idx]
            val_dates = unique_times[val_idx]
            # Contexte : les seq_len heures juste avant la validation
            if val_idx[0] < seq_len:
                print(
                    f"   [fold {fold+1}] Pas assez d'historique, fold ignoré.")
                continue
            ctx_dates = unique_times[val_idx[0] - seq_len: val_idx[0]]

            # --- B. EXTRACTION DES MASQUES ---
            df_train = df[df[self.time_column].isin(train_dates)]
            df_val_raw = df[df[self.time_column].isin(val_dates)]
            df_ctx = df[df[self.time_column].isin(ctx_dates)]

            # --- C. SCALERS (Fit sur Train uniquement) ---
            # On utilise _to_arrays juste pour avoir les chiffres bruts pour le scaler
            X_tr_raw, y_tr_raw, _ = self._to_arrays(df_train, feat_cols, self.predict_column)
            
            scaler_X_fold = StandardScaler().fit(X_tr_raw)
            scaler_y_fold = MinMaxScaler(feature_range=(0, 1)).fit(y_tr_raw.reshape(-1, 1))

            # --- D. GÉNÉRATION DES SÉQUENCES (Le moment de vérité) ---
            X_tr_seq, y_tr_seq = self._prepare_sequences(
                df_train, feat_cols, scaler_X_fold, scaler_y_fold, seq_len)

            # Pour la val, on colle le contexte
            df_val_with_ctx = pd.concat([df_ctx, df_val_raw])
            X_val_seq, y_val_seq = self._prepare_sequences(
                df_val_with_ctx, feat_cols, scaler_X_fold, scaler_y_fold, seq_len)

            # --- LES PRINTS CRITIQUES ---
            print(f"   Dates Train : {train_dates[0]} au {train_dates[-1]}")
            print(f"   Lignes Train brutes : {len(df_train)}")
            print(f"   Séquences LSTM Train: {X_tr_seq.shape[0]}")
            print(f"   Séquences LSTM Val  : {X_val_seq.shape[0]}")

            if len(X_tr_seq) == 0:
                continue

            # Modèle frais pour chaque fold
            net = self._build_dl_model(X_tr_seq.shape[2])
            opt = torch.optim.Adam(net.parameters(), lr=self.DL_LR)
            loss_fn = nn.MSELoss()
            
            print(f"DEBUG SCALING - Fold {fold+1}")
            print(
                f"X_tr_seq MIN: {X_tr_seq.min():.2f} | MAX: {X_tr_seq.max():.2f} | MEAN: {X_tr_seq.mean():.2f}")
            
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_tr_seq),
                            torch.from_numpy(y_tr_seq)),
                batch_size=params["batch"], shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_val_seq),
                            torch.from_numpy(y_val_seq)),
                batch_size=params["batch"], shuffle=False,
            )

            best_val_loss = float("inf")
            best_epoch = 0
            patience_cnt = 0
            patience = getattr(self, "DL_PATIENCE", 20)
            grad_clip = getattr(self, "DL_GRAD_CLIP", 1.0)

            print(f"patience: {patience}")
            print(f"grad clip: {grad_clip}")
            pbar = tqdm(range(1, params["epochs"] + 1), desc=f"Fold {fold+1}")
            for epoch in pbar:
                train_loss = 0.0
                net.train()
                for xb, yb in train_loader:
                    opt.zero_grad()
                    loss = loss_fn(net(xb), yb)
                    loss.backward()
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                    opt.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)
                net.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        val_loss += loss_fn(net(xb), yb).item()
                val_loss /= len(val_loader) 

                pbar.set_postfix({"tr_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}", "best": best_epoch})
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    # On voit l'époque qui gagne
                    pbar.set_description(f"Fold {fold+1} (Best: {epoch})")
                    pbar.refresh()
                    pbar.set_postfix(
                        {"tr_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}", "best": best_epoch})
                    pbar.refresh()
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience:
                        break

            # MAE en unités originales
            net.eval()
            with torch.no_grad():
                preds_sc = net(torch.from_numpy(X_val_seq)).numpy()
            preds = scaler_y_fold.inverse_transform(
                preds_sc.reshape(-1, 1)).ravel()
            truth = scaler_y_fold.inverse_transform(
                y_val_seq.reshape(-1, 1)).ravel()
            fold_mae = float(mean_absolute_error(truth, preds))

            fold_maes.append(fold_mae)
            fold_best_epochs.append(best_epoch)
            print(f"  [{self.model_type}] fold {fold+1} | MAE: {fold_mae:.4f} | best epoch: {best_epoch}")

        cv_mae = float(np.mean(fold_maes)) if fold_maes else float("nan")
        # Médiane des best epochs — plus robuste que la moyenne aux folds extrêmes
        refit_epochs = int(np.median(fold_best_epochs)) if fold_best_epochs else params["epochs"]
        print(f"\n[{self.model_type}] CV MAE: {cv_mae:.4f} ± {np.std(fold_maes):.4f}")
        print(f"[{self.model_type}] Refit sur {refit_epochs} epochs (médiane des best epochs par fold)")

        self.evaluation_results = {
            "cv_mae":      cv_mae,
            "cv_mae_std":  float(np.std(fold_maes)),
            "refit_epochs": refit_epochs,
        }


        # ------------------------------------------------------------------ #
        #  Phase 2 : Refit sur toutes les données train                       #
        #  Scaler refitté sur tout le train, epochs = médiane des best epochs #
        # ------------------------------------------------------------------ #
        # On fitte sur TOUT le train brut
        X_raw, y_raw, _ = self._to_arrays(df, feat_cols, self.predict_column)
        
        if not self.scaler_X or not self.scaler_y:
            raise ValueError("Scalers where not defined properly")
        
        self.scaler_X.fit(X_raw)
        self.scaler_y.fit(y_raw.reshape(-1, 1))

        # ON REUTILISE LA MÊME FONCTION
        X_seq, y_seq = self._prepare_sequences(
            df, feat_cols, self.scaler_X, self.scaler_y, seq_len
        )

        self.input_size = X_seq.shape[2]
        self.model = self._build_dl_model(self.input_size)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.DL_LR)
        loss_fn = nn.MSELoss()
        grad_clip = getattr(self, "DL_GRAD_CLIP", 1.0)

        print(f"--- FINAL REFIT INFO ---")
        print(f"Séquences totales pour entraînement final : {X_seq.shape[0]}")

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq)),
            batch_size=params["batch"], shuffle=True,
        )

        loss_history = {"train": [], "eval": []}

        self.model.train()
        for epoch in tqdm(range(1, refit_epochs + 1)):
            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                opt.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            loss_history["train"].append(avg_loss)

            if epoch % 5 == 0 or epoch == refit_epochs:
                print(
                    f"  [refit] epoch {epoch}/{refit_epochs} | loss: {avg_loss:.6f}")
        print('end')
        # Pas de val loss pendant le refit — on n'a pas de set de val dédié
        return loss_history

    def _predict_deep(self, df: pd.DataFrame) -> np.ndarray:
        import torch
        params = self._get_dl_params()
        seq_len = params["seq_len"]

        # 1. On définit les colonnes de features (en excluant site_name du calcul mais pas du DF)
        feat_cols = [c for c in df.columns if c not in (
            self.time_column,
            self.predict_column,
            'site_name'
        )]
        # print(feat_cols)
        # 2. On utilise notre nouvelle fonction unique !
        # On passe le DF complet. Le scaler_X et scaler_y (fittés au train) sont utilisés.
        # On n'a pas besoin de y réels pour prédire, donc prepare_sequences s'en moque.
        X_seq, y_true_sc = self._prepare_sequences(
            df, feat_cols, self.scaler_X, self.scaler_y, seq_len
        )

        # 3. Inférence PyTorch
        if len(X_seq) == 0:
            return np.array([])

        self.model.eval()
        with torch.no_grad():
            # Conversion en tenseur et prédiction
            preds_sc = self.model(torch.from_numpy(X_seq)).numpy()

        # 4. Retour à l'échelle originale
        preds = self.scaler_y.inverse_transform(
        preds_sc.reshape(-1, 1)).ravel()
        y_true = self.scaler_y.inverse_transform(y_true_sc.reshape(-1, 1)).ravel()
        
        return preds, y_true
