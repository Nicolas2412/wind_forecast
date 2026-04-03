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
                    n = param.size(0)
                    param.data[n//4: n//2].fill_(1.0)  # forget gate bias = 1
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
    import math
    import torch
    import torch.nn as nn

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            x = x + self.pe[:, :x.size(1), :]
            return x

    class _TransformerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            self.pos_encoder = PositionalEncoding(d_model)
            self.dropout = nn.Dropout(dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout,
                dim_feedforward=d_model * 4, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):                       # x: (B, T, F)
            x = self.input_proj(x)                  # (B, T, d_model)
            x = self.pos_encoder(x)
            x = self.dropout(x)
            x = self.encoder(x)                     # (B, T, d_model)
            x = x[:, -1, :]                         # last timestep (B, d_model)
            return self.fc(x).squeeze(-1)            # (B,)

    return _TransformerNet()

# ---------------------------------------------------------------------------
# DataProcessor
# ---------------------------------------------------------------------------

class DataProcessor:
    """Data Processing class."""

    def __init__(self, path_folder: str, X: pd.DataFrame = None, drop_columns: list = ["site_name"]):
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
                    high_thresh: float = 0.90) -> pd.DataFrame:

        df = self.df.copy()

        df["production_normalized"] = df["production"] / df["installed_capacity"]

        df = compute_plateau(df=df, N=N, window=window, tolerance=tolerance, low_thresh=low_thresh, high_thresh=high_thresh)
        
        self.df = df
        return df

    # def impute_production(self, df: pd.DataFrame, max_gap_hours: int = 6) -> pd.DataFrame:
    #     results = []
    #     for site, grp in df.groupby("site_name"):
    #         # On s'assure que le temps est bien au format datetime
    #         grp[self.time_column] = pd.to_datetime(grp[self.time_column])
    #         grp = grp.sort_values(self.time_column).copy()

    #         # 1. Plateaux → NaN
    #         if "is_not_plateau" in grp.columns:
    #             grp.loc[~grp["is_not_plateau"], "production_normalized"] = np.nan

    #         # --- FIX ICI ---
    #         # On passe la colonne temporelle en index pour permettre l'interpolation "time"
    #         grp = grp.set_index(self.time_column)

    #         # 2. Interpolation temporelle
    #         grp["production_normalized"] = (
    #             grp["production_normalized"]
    #             .interpolate(method="time", limit=max_gap_hours)
    #         )

    #         # On repasse en index numérique pour la suite des opérations (fillna, etc.)
    #         grp = grp.reset_index()
    #         # ---------------

    #         # 3. Trous longs résiduels → médiane du site
    #         site_median = grp["production_normalized"].median()
    #         grp["production_normalized"] = grp["production_normalized"].fillna(
    #             site_median)

    #         # 4. ffill/bfill pour les bords
    #         grp["production_normalized"] = (
    #             grp["production_normalized"].ffill().bfill()
    #         )

    #         results.append(grp)

        # return pd.concat(results, ignore_index=True)
    
    def impute_production(self, df: pd.DataFrame, max_gap_hours: int = 6) -> pd.DataFrame:
        results = []
        for site, grp in df.groupby("site_name"):
            grp[self.time_column] = pd.to_datetime(grp[self.time_column])
            grp = grp.sort_values(self.time_column).copy()
            grp[self.time_column] = pd.to_datetime(grp[self.time_column])

            if "is_not_plateau" in grp.columns:
                grp.loc[~grp["is_not_plateau"], "production_normalized"] = np.nan

            # 2. Interpolation temporelle (On garde, c'est le top pour les petits trous)
            grp = grp.set_index(self.time_column)
            grp["production_normalized"] = grp["production_normalized"].interpolate(
                method="time", limit=max_gap_hours

            grp["production_normalized"] = (
                grp["production_normalized"]
                .interpolate(method="time", limit=max_gap_hours)
            )

            grp = grp.reset_index()

            # Au lieu d'une médiane fixe, on utilise la moyenne par heure (0-23h)
            if grp["production_normalized"].isnull().any():
                # On crée une clé temporaire pour l'heure
                grp['tmp_hour'] = grp[self.time_column].dt.hour

                # On calcule la moyenne de prod pour chaque heure sur ce site
                hourly_map = grp.groupby('tmp_hour')[
                    "production_normalized"].mean()

                # Si une heure est totalement vide (rare), on met la médiane du site en secours
                hourly_map = hourly_map.fillna(
                    grp["production_normalized"].median())

                # On remplit les NaNs en mappant l'heure sur nos moyennes
                grp["production_normalized"] = grp["production_normalized"].fillna(
                    grp['tmp_hour'].map(hourly_map)
                )
                grp = grp.drop(columns=['tmp_hour'])
            # --------------------------------------------------

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
        df["month"] = df[self.time_column].dt.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
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
        
        min_lag_h = data_delay_days * 24
        
        results = []
        for site, grp in df.groupby("site_name"):
            grp = grp.set_index(self.time_column).sort_index()

            full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq="1h", tz="UTC")
            grp = grp.reindex(full_idx)
            grp["site_name"] = site

            grp[f"production_lag{min_lag_h}h"] = grp["production_normalized"].shift(
                min_lag_h)
            grp[f"production_lag{min_lag_h + 24}h"] = grp["production_normalized"].shift(
                min_lag_h + 24)
            grp["production_lag720h"] = grp["production_normalized"].shift(720)

            shifted = grp["production_normalized"].shift(min_lag_h)
            grp["production_rolling_mean_7d"] = shifted.rolling(
                168, min_periods=84).mean()
            grp["production_rolling_std_7d"] = shifted.rolling(
                168, min_periods=84).std()
            grp["production_rolling_mean_14d"] = shifted.rolling(
                336, min_periods=168).mean()
            grp["production_rolling_std_14d"] = shifted.rolling(
                336, min_periods=168).std()

            if "is_not_plateau" in grp.columns:
                grp = grp[grp["is_not_plateau"].fillna(False)]

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
        
        df = df.replace([np.inf, -np.inf], np.nan)

        return df
    
    def finalize_for_model(self, df):
        to_exclude = self.drop_columns

        cols_to_keep = [c for c in df.columns if c not in to_exclude]

        mask = df[cols_to_keep].notna().all(axis=1)

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

    def __init__(self, model_type: str = DEFAULT_MODEL_TYPE, savepath:str = None, verbose:bool = False, seq_len: int = 48):
        self.verbose        = verbose
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
        self.LSTM_SEQ_LEN = seq_len
        self.LSTM_HIDDEN = 128
        self.LSTM_LAYERS = 2
        self.LSTM_DROPOUT = 0.4
        self.TRANSFORMER_SEQ_LEN = 48
        self.TRANSFORMER_D_MODEL = 64
        self.TRANSFORMER_NHEAD = 4
        self.TRANSFORMER_LAYERS = 2
        self.TRANSFORMER_DROPOUT = 0.1
        self.DL_EPOCHS = 20
        self.DL_BATCH_SIZE = 256
        self.DL_LR = 5e-5
        self.DL_GRAD_CLIP = 1.0
        self.DL_PATIENCE = 10
        self.model          = None          # built lazily or at train time
        self.scaler_X       = None          # used by LSTM / Transformer
        self.scaler_y       = None          # used by LSTM / Transformer
        self.evaluation_results: dict = {}
        self.savepath = savepath

        self._apply_model_params_from_config()
        
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
                
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, no_cv: bool = False) -> None:
        """Train with time-series cross-validation, then refit on full data."""
        dispatch = {
            "random_forest": self._train_sklearn,
            "xgboost":       self._train_sklearn,
            "lightgbm":      self._train_sklearn,
            "sarimax":       self._train_sarimax,
            "lstm": lambda df: self._train_deep(df, no_cv=no_cv),
            "transformer": lambda df: self._train_deep(df, no_cv=no_cv),
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
                seq_len = self._get_dl_params()["seq_len"]
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
                site_data = df[df['site_name'] == site]
                site_data_len = len(df[df['site_name'] == site])
                n_seq = site_data_len - seq_len

                if n_seq <= 0:
                    continue

                site_pred = prediction[current_idx: current_idx + n_seq]
                site_true = y_true[current_idx: current_idx + n_seq]
                
                site_dates = pd.to_datetime(
                    site_data[self.time_column].iloc[seq_len:].values)

                temp_site = pd.DataFrame({
                    'total_true': site_true.flatten(),
                    'total_pred': site_pred.flatten(),
                    'count': 1
                }, index=site_dates)

                sum_df = sum_df.add(temp_site, fill_value=0)

                s_mae = float(mean_absolute_error(site_true, site_pred))
                s_rmse = float(np.sqrt(mean_squared_error(site_true, site_pred)))
                s_mean_y = np.mean(site_true)
                s_nrmse = float(
                    s_rmse / s_mean_y) if s_mean_y != 0 else 0.0  # Ajout ici


                results["per_site_metrics"][site] = {
                    "mae": s_mae,
                    "rmse": s_rmse,
                    "nrmse": s_nrmse  # Ajout ici
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
                portfolio_mae_total = float(mean_absolute_error(
                    full_portfolio_data['total_true'],
                    full_portfolio_data['total_pred']
                ))
                portfolio_mae_per_site = portfolio_mae_total / n_sites

                portfolio_rmse_total = float(np.sqrt(mean_squared_error(
                    full_portfolio_data['total_true'],
                    full_portfolio_data['total_pred']
                )))
                portfolio_rmse_per_site = portfolio_rmse_total / n_sites

                mean_portfolio_y = np.mean(full_portfolio_data['total_true'])


                portfolio_nrmse_total = float(
                    portfolio_rmse_total / mean_portfolio_y) if mean_portfolio_y != 0 else 0.0

            results.update({
                "portfolio_mae_total": portfolio_mae_total,
                "portfolio_mae_per_site": portfolio_mae_per_site,
                "portfolio_rmse_total": portfolio_rmse_total,
                "portfolio_rmse_per_site": portfolio_rmse_per_site,
                "portfolio_nrmse_total": portfolio_nrmse_total,  # Ajout ici
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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(load_path, map_location=device,
                                    weights_only=False)  # 👈 en premier
            self.input_size = checkpoint['input_size']
            self.scaler_X = checkpoint['scaler_X']
            self.scaler_y = checkpoint['scaler_y']
            self.model = self._build_dl_model(self.input_size)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(device)
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
        if self.model is None:
            self.model = self._build_sklearn_model()

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        maes = []
        
        unique_times = np.sort(df[self.time_column].unique())
        
        for train_idx, val_idx in tscv.split(unique_times):
            train_dates = unique_times[train_idx]
            val_dates = unique_times[val_idx]
            
            df_tr = df[df[self.time_column].isin(train_dates)]
            df_val = df[df[self.time_column].isin(val_dates)]
            
            X_tr = df_tr.drop(columns=[self.time_column, self.predict_column, 'site_name'], errors='ignore')
            y_tr = df_tr[self.predict_column]
            X_val = df_val.drop(columns=[self.time_column, self.predict_column, 'site_name'], errors='ignore')
            y_val = df_val[self.predict_column]

            fold_model = clone(self.model)
            fold_model.fit(X_tr, y_tr)
            maes.append(mean_absolute_error(y_val, fold_model.predict(X_val)))

        cv_mae = float(np.mean(maes))
        self.evaluation_results = {"cv_mae": cv_mae, "cv_mae_std": float(np.std(maes))}
        print(f"[{self.model_type}] CV MAE: {cv_mae:.4f} ± {np.std(maes):.4f}")

        X_full = df.drop(columns=[self.time_column, self.predict_column, 'site_name'], errors='ignore')
        y_full = df[self.predict_column]
        self.model.fit(X_full, y_full)

    def _predict_sklearn(self, df: pd.DataFrame) -> np.ndarray:
        X = df.drop(columns=[self.time_column, self.predict_column, 'site_name'], errors="ignore")
        if X.empty:
            raise ValueError("No prediction features available after preprocessing.")
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
        print(seq_len)
        all_X, all_y = [], []
        
        for site, grp in df.groupby("site_name"):
            if len(grp) <= seq_len:
                continue

            X_site_raw = grp[feat_cols].values.astype(np.float32)
            y_site_raw = grp[self.predict_column].values.astype(
                np.float32).reshape(-1, 1)
            
            cols_to_scale = [
                c for c in feat_cols if not c.endswith(('sin', 'cos'))]
            idx_to_scale = [feat_cols.index(c) for c in cols_to_scale]
            idx_no_scale = [feat_cols.index(c)
                            for c in feat_cols if c.endswith(('sin', 'cos'))]

            X_sc_part = scaler_X.transform(X_site_raw[:, idx_to_scale])
            X_sc_part = np.clip(X_sc_part, -5.0, 5.0)
            
            X_site_sc = np.zeros_like(X_site_raw)
            X_site_sc[:, idx_to_scale] = X_sc_part
            X_site_sc[:, idx_no_scale] = X_site_raw[:, idx_no_scale]
            
            is_clipped = np.abs(X_site_sc) > 5.0
            pct_clipped_global = np.mean(is_clipped) * 100

            if pct_clipped_global > 1.0:
                print(
                    f"  ⚠️ [{site}] {pct_clipped_global:.2f}% des valeurs globales clippées. Détails :")
                # On calcule le pourcentage de clipping spécifiquement pour chaque colonne
                pct_per_feature = np.mean(is_clipped, axis=0) * 100

                for i, col_name in enumerate(feat_cols):
                    # Affiche toute variable qui subit un clipping
                    if pct_per_feature[i] > 0.0:
                        print(
                            f"      -> {col_name} : {pct_per_feature[i]:.2f}% clippé")

            
            X_site_sc = np.clip(X_site_sc, -5.0, 5.0) #Pour etre sur de pas avoir de valeur aberantes
            y_site_sc = scaler_y.transform(y_site_raw).ravel()
            
            
            # 3. Fenêtrage glissante
            for i in range(len(X_site_sc) - seq_len):
                all_X.append(X_site_sc[i: i + seq_len])
                all_y.append(y_site_sc[i + seq_len])

        if not all_X:
            return np.array([]), np.array([])

        return np.array(all_X), np.array(all_y)

    def _train_deep(self, df: pd.DataFrame, no_cv=False):
        import copy
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{self.model_type}] Utilisation de : {device}")


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
        tscv = TimeSeriesSplit(n_splits=self.n_splits, gap=seq_len)

        print(f"NO CV: {no_cv}")
        if not no_cv:
            fold_maes, fold_best_epochs = [], []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(unique_times)):
                print(f"\n--- Cross validation Fold {fold+1}/{self.n_splits} ---")

                train_dates = unique_times[train_idx]
                val_dates = unique_times[val_idx]
                # Contexte : les seq_len heures juste avant la validation
                if val_idx[0] < seq_len:
                    print(
                        f"   [fold {fold+1}] Pas assez d'historique, fold ignoré.")
                    continue
                ctx_dates = unique_times[val_idx[0] - seq_len: val_idx[0]]

                df_train = df[df[self.time_column].isin(train_dates)]
                df_val_raw = df[df[self.time_column].isin(val_dates)]
                df_ctx = df[df[self.time_column].isin(ctx_dates)]

                X_tr_raw, y_tr_raw, _ = self._to_arrays(
                    df_train, feat_cols, self.predict_column)


                idx_to_scale = [i for i, c in enumerate(
                    feat_cols) if not c.endswith(('sin', 'cos'))]

                scaler_X_fold = StandardScaler().fit(X_tr_raw[:, idx_to_scale])
                
                scaler_y_fold = MinMaxScaler(feature_range=(0, 1)).fit(y_tr_raw.reshape(-1, 1))

                X_tr_seq, y_tr_seq = self._prepare_sequences(
                    df_train, feat_cols, scaler_X_fold, scaler_y_fold, seq_len)

                df_val_with_ctx = pd.concat([df_ctx, df_val_raw])
                X_val_seq, y_val_seq = self._prepare_sequences(
                    df_val_with_ctx, feat_cols, scaler_X_fold, scaler_y_fold, seq_len)

                if self.verbose:
                    print(f"   Dates Train : {train_dates[0]} au {train_dates[-1]}")
                    print(f"   Lignes Train brutes : {len(df_train)}")
                    print(f"   Séquences LSTM Train: {X_tr_seq.shape[0]}")
                    print(f"   Séquences LSTM Val  : {X_val_seq.shape[0]}")

                if len(X_tr_seq) == 0:
                    continue

                # Modèle frais pour chaque fold
                net = self._build_dl_model(X_tr_seq.shape[2]).to(device)
                opt = torch.optim.Adam(
                    net.parameters(), lr=self.DL_LR, weight_decay=1e-5)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode='min', patience=5, factor=0.5, min_lr=1e-6
                )
                loss_fn = nn.MSELoss()
                
                if self.verbose:      
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
                best_state = None
                patience = getattr(self, "DL_PATIENCE", 20)
                grad_clip = getattr(self, "DL_GRAD_CLIP", 1.0)

                if self.verbose:
                    print(f"patience: {patience}")
                    print(f"grad clip: {grad_clip}")
                pbar = tqdm(range(1, params["epochs"] + 1), desc=f"Fold {fold+1}", disable=not self.verbose)
                for epoch in pbar:
                    train_loss = 0.0

                    train_preds_list = []
                    train_true_list = []
                    net.train()
                    for xb, yb in train_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        opt.zero_grad()
                        pred = net(xb)
                        loss = loss_fn(pred, yb)
                        loss.backward()
                        if grad_clip > 0:
                            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                        opt.step()
                        train_loss += loss.item()

                        train_preds_list.append(pred.detach().cpu().numpy())
                        train_true_list.append(yb.cpu().numpy())
                        
                    train_loss /= len(train_loader)

                    train_preds_all = scaler_y_fold.inverse_transform(
                        np.concatenate(train_preds_list).reshape(-1, 1)).ravel()
                    train_true_all = scaler_y_fold.inverse_transform(
                        np.concatenate(train_true_list).reshape(-1, 1)).ravel()
                    tr_mae = mean_absolute_error(train_true_all, train_preds_all)
                    
                    net.eval()
                    val_loss = 0.0
                    val_preds_list = []
                    val_true_list = []

                    with torch.no_grad():
                        for xb, yb in val_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            pred = net(xb)
                            val_loss += loss_fn(pred, yb).item()
                            val_preds_list.append(pred.cpu().numpy())
                            val_true_list.append(yb.cpu().numpy())
                    val_loss /= len(val_loader) 
                    val_preds_all = scaler_y_fold.inverse_transform(
                        np.concatenate(val_preds_list).reshape(-1, 1)).ravel()
                    val_true_all = scaler_y_fold.inverse_transform(
                        np.concatenate(val_true_list).reshape(-1, 1)).ravel()
                    val_mae = mean_absolute_error(val_true_all, val_preds_all)
                    scheduler.step(val_loss)
                    
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        best_state = copy.deepcopy(net.state_dict())
                        # On voit l'époque qui gagne
                        pbar.set_description(f"Fold {fold+1} (Best: {epoch})")
                        pbar.refresh()
                        pbar.set_postfix({
                            "tr_loss": f"{train_loss:.4f}",
                            "val_loss": f"{val_loss:.4f}",
                            "tr_mae": f"{tr_mae:.4f}",
                            "val_mae": f"{val_mae:.4f}",
                            "best": best_epoch
                        })

                        pbar.refresh()
                        patience_cnt = 0
                    else:
                        patience_cnt += 1
                        if patience_cnt >= patience:
                            break

                # Loading the best model for this fold to evaluate on the val set
                if best_state is not None:
                    net.load_state_dict(best_state)
                net.eval()
                preds_chunks = []
                with torch.no_grad():

                    for i in range(0, len(X_val_seq), params["batch"]):
                        chunk = torch.from_numpy(X_val_seq[i:i+params["batch"]]).to(device)
                        preds_chunks.append(net(chunk).cpu().numpy())
                        preds_sc = np.concatenate(preds_chunks)

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
            refit_epochs = refit_epochs = int(np.median(fold_best_epochs[-3:])) if fold_best_epochs else params["epochs"]
            print(f"\n[{self.model_type}] CV MAE: {cv_mae:.4f} ± {np.std(fold_maes):.4f}")
            print(f"[{self.model_type}] Refit sur {refit_epochs} epochs (médiane des best epochs par fold)")

            self.evaluation_results = {
                "cv_mae":      cv_mae,
                "cv_mae_std":  float(np.std(fold_maes)),
                "refit_epochs": refit_epochs,
            }
        
        else:
            refit_epochs = self.DL_EPOCHS
            self.evaluation_results = {"cv_mae": float(
                "nan"), "cv_mae_std": float("nan"), "refit_epochs": refit_epochs}
            print(f"[{self.model_type}] CV désactivé, refit sur {refit_epochs} epochs")


        X_raw, y_raw, _ = self._to_arrays(df, feat_cols, self.predict_column)


        if not self.scaler_X or not self.scaler_y:
            raise ValueError("Scalers where not defined properly")

        idx_to_scale = [i for i, c in enumerate(
            feat_cols) if not c.endswith(('sin', 'cos'))]

        self.scaler_X.fit(X_raw[:, idx_to_scale])
        self.scaler_y.fit(y_raw.reshape(-1, 1))

        X_seq, y_seq = self._prepare_sequences(
            df, feat_cols, self.scaler_X, self.scaler_y, seq_len
        )

        self.input_size = X_seq.shape[2]
        self.model = self._build_dl_model(self.input_size).to(device)
        opt = torch.optim.Adam(self.model.parameters(),
                               lr=self.DL_LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', patience=5, factor=0.5, min_lr=1e-6
        )
        loss_fn = nn.MSELoss()
        grad_clip = getattr(self, "DL_GRAD_CLIP", 1.0)

        print(f"--- FINAL REFIT INFO ---")
        print(f"Séquences totales pour entraînement final : {X_seq.shape[0]}")

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq)),
            batch_size=params["batch"], 
            shuffle=True,
            num_workers=4,      # parallélise le chargement
            pin_memory=True,    # accélère le transfert CPU→GPU si GPU dispo
            persistent_workers=True  # évite de recréer les workers à chaque epoch
            
        )

        loss_history = {"train": [], "eval": []}

        self.model.train()
        for epoch in tqdm(range(1, refit_epochs + 1), disable=not self.verbose):
            epoch_loss = 0.0
            preds_list = []
            true_list = []
            for xb, yb in tqdm(loader, desc=f"  epoch {epoch}", leave=False, disable=not self.verbose):
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                opt.step()
                epoch_loss += loss.item()
                preds_list.append(pred.detach().cpu().numpy())
                true_list.append(yb.cpu().numpy())
            avg_loss = epoch_loss / len(loader)
            scheduler.step(avg_loss)
            loss_history["train"].append(avg_loss)

            tr_preds = self.scaler_y.inverse_transform(
                np.concatenate(preds_list).reshape(-1, 1)).ravel()
            tr_true = self.scaler_y.inverse_transform(
                np.concatenate(true_list).reshape(-1, 1)).ravel()
            tr_mae = mean_absolute_error(tr_true, tr_preds)
            if self.verbose:
                print(f"  [refit] epoch {epoch}/{refit_epochs} | loss: {avg_loss:.6f} | MAE train: {tr_mae:.4f}")

        print('end')
        # Pas de val loss pendant le refit — on n'a pas de set de val dédié
        return loss_history

    def _predict_deep(self, df: pd.DataFrame) -> np.ndarray:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params = self._get_dl_params()
        seq_len = params["seq_len"]

        feat_cols = [c for c in df.columns if c not in (
            self.time_column,
            self.predict_column,
            'site_name'
        )]
        print(feat_cols)
        X_seq, y_true_sc = self._prepare_sequences(
            df, feat_cols, self.scaler_X, self.scaler_y, seq_len
        )

        if len(X_seq) == 0:
            return np.array([])

        self.model.eval()
        self.model.to(device)
        preds_chunks = []
        with torch.no_grad():
            for i in range(0, len(X_seq), 2048):
                chunk = torch.from_numpy(X_seq[i:i+2048]).to(device)
                preds_chunks.append(self.model(chunk).cpu().numpy())
        preds_sc = np.concatenate(preds_chunks)

        preds = self.scaler_y.inverse_transform(
        preds_sc.reshape(-1, 1)).ravel()
        y_true = self.scaler_y.inverse_transform(y_true_sc.reshape(-1, 1)).ravel()
        
        return preds, y_true
