"""
Preprocessing Module for SIH26081 — Hybrid Weather Forecast Blending System.

Performs feature engineering (cyclical time encoding, categorical encoding),
feature scaling (StandardScaler), and enforces strict zero-target-leakage checks.
"""

import os
import sys

# Ensure project root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from ml import config



class WeatherPreprocessor:
    """Preprocesses weather data features while ensuring strict isolation of reference rainfall."""

    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.regime_categories: List[str] = ["Monsoon", "Winter", "Summer", "Transition"]

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineers cyclical temporal features and one-hot encodes weather regimes."""
        df_feats = df.copy()

        # Cyclical Encodings for Month and Day of Year
        month_rad = 2 * np.pi * df_feats["month"] / 12.0
        df_feats["month_sin"] = np.sin(month_rad)
        df_feats["month_cos"] = np.cos(month_rad)

        day_rad = 2 * np.pi * df_feats["day_of_year"] / 365.25
        df_feats["day_sin"] = np.sin(day_rad)
        df_feats["day_cos"] = np.cos(day_rad)

        # One-Hot Encoding for Weather Regime
        for cat in self.regime_categories:
            df_feats[f"regime_{cat.lower()}"] = (df_feats["weather_regime"] == cat).astype(float)

        return df_feats

    def get_feature_columns(self) -> List[str]:
        """Returns the complete list of engineered input feature names."""
        engineered = [
            "month_sin", "month_cos", "day_sin", "day_cos"
        ] + [f"regime_{cat.lower()}" for cat in self.regime_categories]
        
        return config.RAW_NUMERICAL_FEATURES + engineered

    def verify_no_target_leakage(self, feature_df: pd.DataFrame) -> None:
        """
        CRITICAL SCIENTIFIC REQUIREMENT:
        Ensures target reference_rainfall_mm is strictly excluded from features.
        """
        forbidden = [config.REFERENCE_TARGET_COL, "reference_rain", "target"]
        for col in feature_df.columns:
            if col in forbidden:
                raise ValueError(
                    f"CRITICAL TARGET LEAKAGE DETECTED: Feature matrix contains target column '{col}'!"
                )

    def fit_transform(self, train_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Fits scaler on training data and transforms training features."""
        df_engineered = self.transform_features(train_df)
        self.feature_names = self.get_feature_columns()

        X_raw = df_engineered[self.feature_names]
        self.verify_no_target_leakage(X_raw)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)

        return X_scaled, self.feature_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms validation or test data using fitted scaler."""
        if self.scaler is None:
            raise RuntimeError("Preprocessor has not been fitted yet! Call fit_transform first.")

        df_engineered = self.transform_features(df)
        X_raw = df_engineered[self.feature_names]
        self.verify_no_target_leakage(X_raw)

        return self.scaler.transform(X_raw)

    def save(self, path: str = config.SCALER_PKL_PATH) -> None:
        """Saves scaler state to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"scaler": self.scaler, "feature_names": self.feature_names}, path)
        print(f"💾 Preprocessor scaler saved to '{path}'")

    def load(self, path: str = config.SCALER_PKL_PATH) -> None:
        """Loads scaler state from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at '{path}'")
        data = joblib.load(path)
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
