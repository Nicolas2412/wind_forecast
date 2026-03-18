import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone

class DataProcessor:
    """
    Data Processing class
    """
    def __init__(self, path_folder: str, X: pd.DataFrame = None):
        self.path = path_folder
        self.time_column = "delivery_time"
        self.predict_column = "production_normalized"
        self.df = self.open_data() if X is None else X


    def open_data(self) -> pd.DataFrame:
        """Open data from path"""
        df = pd.DataFrame()
        for file in Path(self.path).glob("*.parquet"):
            if file.name.startswith("dataset_2"):
                continue
            df = pd.concat([df, pd.read_parquet(file)], ignore_index=True)
        return df

    def engineer_features(self) -> pd.DataFrame:
        """Engineer features for meta-model"""
        df = self.df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        df["hour"] = df[self.time_column].dt.hour
        df["day_of_week"] = df[self.time_column].dt.dayofweek
        df["month"] = df[self.time_column].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        df['production_normalized'] = df['production'] / df['installed_capacity']
        df.drop(columns=["production", "installed_capacity"], inplace=True)
        df["wind_speed_diff"] = df["wind_speed_100m"] - df["wind_speed_10m"]
        for col in ["wind_speed_10m", "wind_speed_100m"]:
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_cubed'] = df[col] ** 3
        df['wind_speed_ratio'] = df['wind_speed_100m'] / (df['wind_speed_10m'] + 1e-8)

        # statistical features of production carefull of data leakage
        df['production_lag1'] = df['production_normalized'].shift(1)
        df['production_lag24'] = df['production_normalized'].shift(24)
        df['production_rolling_mean_24'] = df['production_normalized'].rolling(window=24).mean()
        df['production_rolling_std_24'] = df['production_normalized'].rolling(window=24).std()
        df['production_rolling_mean_168'] = df['production_normalized'].rolling(window=168).mean()
        df['production_rolling_std_168'] = df['production_normalized'].rolling(window=168).std()

        return df
    
class ForecastModel:
    """
    Forecasting model class
    """
    def __init__(self, model_type: str = "random_forest"):
        self.model = None
        self.time_column = "delivery_time"
        self.predict_column = "production_normalized"
        self.n_splits = 5
        self.model_type = self.model_validate(model_type)

    def model_validate(self, model_type: str) -> str:
        """Validate model type"""
        valid_models = ["random_forest", "xgboost", "lightgbm", "sarimax", "lstm", 'transformer']
        if model_type not in valid_models:
            raise ValueError(f"Invalid model type. Choose from {valid_models}")
        return model_type

    def train(self, df : pd.DataFrame):
        """Train the model"""
        X_train, y_train = df.drop(columns=[self.time_column, self.predict_column]), df[self.predict_column]
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        maes = []
        for train_idx, val_idx in tscv.split(X_train):
            X_train, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model_clone = clone(self.model)
            model_clone.fit(X_train, y_train)
            preds = model_clone.predict(X_val)
            maes.append(mean_absolute_error(y_val, preds))

        cv_mae = np.mean(maes)
        print(f"Cross-validated MAE: {cv_mae:.4f}")

        self.model.fit(X_train, y_train)


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        X = DataProcessor(X).engineer_features()
        return self.model.predict(X)
