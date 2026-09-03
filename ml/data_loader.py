"""
Data Loader Module for SIH26081 — Hybrid Weather Forecast Blending System.

Handles dataset loading, verification, and strictly chronological train/validation/test splitting
to ensure no future temporal information leaks into past model training.
"""

import os
import sys

# Ensure project root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Tuple, Dict
import pandas as pd
import numpy as np

from ml import config


REQUIRED_COLUMNS = [
    "date", "latitude", "longitude", "month", "day_of_year", "lead_time_hours",
    "temperature_2m_c", "humidity_pct", "surface_pressure_hpa", "u_wind_10m_ms",
    "v_wind_10m_ms", "weather_regime", "nwp_rainfall_mm", "ai_rainfall_mm",
    "ensemble_mean_rainfall_mm", "ensemble_std_rainfall_mm", "nwp_historical_rmse",
    "ai_historical_rmse", "ensemble_historical_rmse", "reference_rainfall_mm"
]


class WeatherDataLoader:
    """Modular data loader adhering to strict time-series weather boundaries."""

    def __init__(self, data_path: str = config.DATA_PATH):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:

        """Loads dataset from CSV or real data source, converts date, sorts chronologically."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset not found at '{self.data_path}'. Please run 'python generate_sample_data.py' first."
            )

        df = pd.read_csv(self.data_path)
        self.validate_schema(df)

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by="date").reset_index(drop=True)
        return df

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Validates that all required columns are present in the dataframe."""
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in dataset: {missing}")

    def get_chronological_splits(
        self,
        df: pd.DataFrame,
        train_end: str = config.TRAIN_END_DATE,
        val_end: str = config.VAL_END_DATE,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits data chronologically:
        - Train: date <= train_end
        - Validation: train_end < date <= val_end
        - Test: date > val_end
        """
        train_mask = df["date"] <= pd.to_datetime(train_end)
        val_mask = (df["date"] > pd.to_datetime(train_end)) & (df["date"] <= pd.to_datetime(val_end))
        test_mask = df["date"] > pd.to_datetime(val_end)

        df_train = df[train_mask].copy().reset_index(drop=True)
        df_val = df[val_mask].copy().reset_index(drop=True)
        df_test = df[test_mask].copy().reset_index(drop=True)

        print(f"📊 Dataset Chronological Split Summary:")
        print(f"   • Train Set:      {len(df_train):,} samples ({df_train['date'].min().strftime('%Y-%m-%d')} to {df_train['date'].max().strftime('%Y-%m-%d')})")
        print(f"   • Validation Set: {len(df_val):,} samples ({df_val['date'].min().strftime('%Y-%m-%d')} to {df_val['date'].max().strftime('%Y-%m-%d')})")
        print(f"   • Test Set:       {len(df_test):,} samples ({df_test['date'].min().strftime('%Y-%m-%d')} to {df_test['date'].max().strftime('%Y-%m-%d')})")

        return df_train, df_val, df_test


if __name__ == "__main__":
    loader = WeatherDataLoader()
    df = loader.load_data()
    train, val, test = loader.get_chronological_splits(df)
