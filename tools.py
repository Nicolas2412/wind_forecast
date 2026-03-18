import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# PyTorch helpers (imported lazily inside the classes that need them)
# ---------------------------------------------------------------------------

def _build_lstm_net(input_size: int, hidden_size: int, num_layers: int, dropout: float):
    """Build a PyTorch LSTM regressor."""
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

        def forward(self, x):                       # x: (B, T, F)
            out, _ = self.lstm(x)                   # (B, T, H)
            out = self.dropout(out[:, -1, :])        # last timestep
            return self.fc(out).squeeze(-1)          # (B,)

    return _LSTMNet()


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
# Sequence builder shared by LSTM / Transformer
# ---------------------------------------------------------------------------

def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Return (Xs, ys) arrays of shape (N, seq_len, F) and (N,)."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len: i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ---------------------------------------------------------------------------
# DataProcessor
# ---------------------------------------------------------------------------

class DataProcessor:
    """Data Processing class."""

    def __init__(self, path_folder: str, X: pd.DataFrame = None):
        self.path = path_folder
        self.time_column = "delivery_time"
        self.predict_column = "production_normalized"
        self.df = self.open_data() if X is None else X

    def open_data(self) -> pd.DataFrame:
        """Open data from path."""
        df = pd.DataFrame()
        for file in Path(self.path).glob("*.parquet"):
            if file.name.startswith("dataset_2"):
                continue
            df = pd.concat([df, pd.read_parquet(file)], ignore_index=True)
        return df

    def prepocess_data(self) -> pd.DataFrame:
        pass

    def engineer_features(self) -> pd.DataFrame:
        """Engineer features for the model."""
        df = self.df.copy()
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

        df["production_normalized"] = df["production"] / df["installed_capacity"]
        df.drop(columns=["production", "installed_capacity"], inplace=True)

        df["wind_speed_diff"]  = df["wind_speed_100m"] - df["wind_speed_10m"]
        for col in ["wind_speed_10m", "wind_speed_100m"]:
            df[f"{col}_squared"] = df[col] ** 2
            df[f"{col}_cubed"]   = df[col] ** 3
        df["wind_speed_ratio"] = df["wind_speed_100m"] / (df["wind_speed_10m"] + 1e-8)

        # Lag / rolling features — be careful of data leakage
        df["production_lag1"]              = df["production_normalized"].shift(1)
        df["production_lag24"]             = df["production_normalized"].shift(24)
        df["production_rolling_mean_24"]   = df["production_normalized"].rolling(24).mean()
        df["production_rolling_std_24"]    = df["production_normalized"].rolling(24).std()
        df["production_rolling_mean_168"]  = df["production_normalized"].rolling(168).mean()
        df["production_rolling_std_168"]   = df["production_normalized"].rolling(168).std()

        return df
    
    def run(self) -> pd.DataFrame:
        """Run the full processing pipeline."""
        self.prepocess_data()
        self.engineer_features()
        return self.df.copy()


# ---------------------------------------------------------------------------
# ForecastModel
# ---------------------------------------------------------------------------

class ForecastModel:
    """
    Unified forecasting model supporting:
        random_forest | xgboost | lightgbm | sarimax | lstm | transformer
    """

    # ----- hyper-parameters exposed as class-level defaults ----------------
    SKLEARN_DEFAULTS = dict(n_estimators=200, random_state=42, n_jobs=-1)
    SARIMAX_ORDER       = (1, 1, 1)
    SARIMAX_SEAS_ORDER  = (1, 1, 1, 24)          # hourly seasonality
    LSTM_SEQ_LEN        = 48
    LSTM_HIDDEN         = 128
    LSTM_LAYERS         = 2
    LSTM_DROPOUT        = 0.2
    TRANSFORMER_SEQ_LEN = 48
    TRANSFORMER_D_MODEL = 64
    TRANSFORMER_NHEAD   = 4
    TRANSFORMER_LAYERS  = 2
    TRANSFORMER_DROPOUT = 0.1
    DL_EPOCHS           = 30
    DL_BATCH_SIZE       = 256
    DL_LR               = 1e-3
    # -----------------------------------------------------------------------

    VALID_MODELS = ["random_forest", "xgboost", "lightgbm", "sarimax", "lstm", "transformer"]

    def __init__(self, model_type: str = "random_forest"):
        self.time_column    = "delivery_time"
        self.predict_column = "production_normalized"
        self.n_splits       = 5
        self.model_type     = self._validate(model_type)
        self.model          = None          # built lazily or at train time
        self.scaler_X       = None          # used by LSTM / Transformer
        self.scaler_y       = None          # used by LSTM / Transformer
        self.evaluation_results: dict = {}

        # Build sklearn / tree models immediately (lightweight)
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

    def evaluate(self, df: pd.DataFrame = None) -> dict:
        """Return stored cross-validation metrics."""
        prediction = self.predict(df.drop(columns=[self.time_column, self.predict_column])) if df is not None else None
        error = mean_absolute_error(df[self.predict_column], prediction) if prediction is not None else None
        return error

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict on a new DataFrame (raw, before feature engineering)."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        processed = DataProcessor(path_folder="", X=df).engineer_features()
        dispatch = {
            "random_forest": self._predict_sklearn,
            "xgboost":       self._predict_sklearn,
            "lightgbm":      self._predict_sklearn,
            "sarimax":       self._predict_sarimax,
            "lstm":          self._predict_deep,
            "transformer":   self._predict_deep,
        }
        return dispatch[self.model_type](processed)

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------

    def _validate(self, model_type: str) -> str:
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
                tree_method="hist",
            )
        if self.model_type == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=self.SKLEARN_DEFAULTS["n_estimators"],
                random_state=self.SKLEARN_DEFAULTS["random_state"],
                n_jobs=self.SKLEARN_DEFAULTS["n_jobs"],
                verbose=-1,
            )

    # ------------------------------------------------------------------
    # sklearn / tree training  (fixes the variable-shadowing bug)
    # ------------------------------------------------------------------

    def _train_sklearn(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=[self.time_column, self.predict_column])
        y = df[self.predict_column]

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        maes = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]   # ← fixed shadowing
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            fold_model = clone(self.model)
            fold_model.fit(X_tr, y_tr)
            maes.append(mean_absolute_error(y_val, fold_model.predict(X_val)))

        cv_mae = float(np.mean(maes))
        self.evaluation_results = {"cv_mae": cv_mae, "cv_mae_std": float(np.std(maes))}
        print(f"[{self.model_type}] CV MAE: {cv_mae:.4f} ± {np.std(maes):.4f}")

        # Final fit on the complete dataset
        self.model.fit(X, y)

    def _predict_sklearn(self, df: pd.DataFrame) -> np.ndarray:
        X = df.drop(columns=[self.time_column, self.predict_column], errors="ignore")
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # SARIMAX
    # ------------------------------------------------------------------

    def _train_sarimax(self, df: pd.DataFrame) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        series = df.set_index(self.time_column)[self.predict_column].asfreq("h")

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
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                preds = res.forecast(steps=len(val_s))
                maes.append(mean_absolute_error(val_s, preds))
            except Exception as exc:
                print(f"  SARIMAX fold skipped: {exc}")

        cv_mae = float(np.mean(maes)) if maes else float("nan")
        self.evaluation_results = {"cv_mae": cv_mae}
        print(f"[sarimax] CV MAE: {cv_mae:.4f}")

        # Final fit on full series
        self.model = SARIMAX(
            series,
            order=self.SARIMAX_ORDER,
            seasonal_order=self.SARIMAX_SEAS_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    def _predict_sarimax(self, df: pd.DataFrame) -> np.ndarray:
        n_steps = len(df)
        forecast = self.model.forecast(steps=n_steps)
        return forecast.values

    # ------------------------------------------------------------------
    # Deep-learning helpers (LSTM & Transformer share the same loop)
    # ------------------------------------------------------------------

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

    def _train_deep(self, df: pd.DataFrame) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        params  = self._get_dl_params()
        seq_len = params["seq_len"]

        # ---- prepare raw arrays -----------------------------------------
        feat_cols = [c for c in df.columns if c not in (self.time_column, self.predict_column)]
        df_clean  = df.dropna(subset=feat_cols + [self.predict_column])

        X_raw = df_clean[feat_cols].values.astype(np.float32)
        y_raw = df_clean[self.predict_column].values.astype(np.float32)

        # ---- scalers (fit on train, reused at inference) -----------------
        self.scaler_X = StandardScaler().fit(X_raw)
        self.scaler_y = StandardScaler().fit(y_raw.reshape(-1, 1))
        X_sc = self.scaler_X.transform(X_raw)
        y_sc = self.scaler_y.transform(y_raw.reshape(-1, 1)).ravel()

        # ---- time-series CV ----------------------------------------------
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        maes = []
        idx  = np.arange(len(X_sc))

        for fold, (train_idx, val_idx) in enumerate(tscv.split(idx)):
            X_tr, y_tr = _make_sequences(X_sc[train_idx], y_sc[train_idx], seq_len)
            X_val_s, y_val_s = _make_sequences(X_sc[val_idx], y_sc[val_idx], seq_len)
            if len(X_tr) == 0 or len(X_val_s) == 0:
                continue

            net = self._build_dl_model(X_tr.shape[2])
            opt = torch.optim.Adam(net.parameters(), lr=self.DL_LR)
            loss_fn = nn.MSELoss()
            loader  = DataLoader(
                TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                batch_size=params["batch"], shuffle=False,
            )
            net.train()
            for _ in range(params["epochs"]):
                for xb, yb in loader:
                    opt.zero_grad()
                    loss_fn(net(xb), yb).backward()
                    opt.step()

            net.eval()
            with torch.no_grad():
                preds_sc = net(torch.from_numpy(X_val_s)).numpy()
            preds = self.scaler_y.inverse_transform(preds_sc.reshape(-1, 1)).ravel()
            truth = self.scaler_y.inverse_transform(y_val_s.reshape(-1, 1)).ravel()
            maes.append(mean_absolute_error(truth, preds))
            print(f"  [{self.model_type}] fold {fold+1} MAE: {maes[-1]:.4f}")

        cv_mae = float(np.mean(maes)) if maes else float("nan")
        self.evaluation_results = {"cv_mae": cv_mae, "cv_mae_std": float(np.std(maes))}
        print(f"[{self.model_type}] CV MAE: {cv_mae:.4f}")

        # ---- final training on full data ---------------------------------
        X_full, y_full = _make_sequences(X_sc, y_sc, seq_len)
        self.model = self._build_dl_model(X_full.shape[2])
        opt     = torch.optim.Adam(self.model.parameters(), lr=self.DL_LR)
        loss_fn = nn.MSELoss()
        loader  = DataLoader(
            TensorDataset(torch.from_numpy(X_full), torch.from_numpy(y_full)),
            batch_size=params["batch"], shuffle=False,
        )
        self.model.train()
        for epoch in range(params["epochs"]):
            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                l = loss_fn(self.model(xb), yb)
                l.backward()
                opt.step()
                epoch_loss += l.item()
            if (epoch + 1) % 10 == 0:
                print(f"  [{self.model_type}] epoch {epoch+1}/{params['epochs']} "
                      f"loss: {epoch_loss/len(loader):.6f}")

    def _predict_deep(self, df: pd.DataFrame) -> np.ndarray:
        import torch

        seq_len   = (self.LSTM_SEQ_LEN if self.model_type == "lstm"
                     else self.TRANSFORMER_SEQ_LEN)
        feat_cols = [c for c in df.columns
                     if c not in (self.time_column, self.predict_column)]

        X_raw = df[feat_cols].values.astype(np.float32)
        X_sc  = self.scaler_X.transform(X_raw)

        y_dummy = np.zeros(len(X_sc), dtype=np.float32)
        X_seq, _ = _make_sequences(X_sc, y_dummy, seq_len)

        self.model.eval()
        with torch.no_grad():
            preds_sc = self.model(torch.from_numpy(X_seq)).numpy()

        return self.scaler_y.inverse_transform(preds_sc.reshape(-1, 1)).ravel()