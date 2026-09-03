"""
Configuration module for SIH26081 Hybrid Weather Forecast Blending System.
Centralizes paths, feature names, train/val/test split thresholds, and hyper-parameters.
"""

import os
from typing import List, Dict, Any

# Project Directory Base
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# File Paths
DATA_PATH = os.path.join(BASE_DIR, "data", "SIH26081_ML_training_sample.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

MODEL_PKL_PATH = os.path.join(MODELS_DIR, "blending_model.pkl")
SCALER_PKL_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
CONFIG_JSON_PATH = os.path.join(MODELS_DIR, "config.json")

# Chronological Split Thresholds (10-Year Dataset: 2015-2024)
TRAIN_END_DATE = "2022-12-31"  # 2015 - 2022 (8 Years Training Data)
VAL_END_DATE = "2023-12-31"    # 2023 (1 Year Validation Data)
TEST_START_DATE = "2024-01-01" # 2024 (1 Year Holdout Test Data)

# Feature Specification
RAW_NUMERICAL_FEATURES: List[str] = [
    "latitude",
    "longitude",
    "lead_time_hours",
    "temperature_2m_c",
    "humidity_pct",
    "surface_pressure_hpa",
    "u_wind_10m_ms",
    "v_wind_10m_ms",
    "ensemble_std_rainfall_mm",
    "nwp_historical_rmse",
    "ai_historical_rmse",
    "ensemble_historical_rmse",
]

RAW_CATEGORICAL_FEATURES: List[str] = [
    "weather_regime"
]

CYCLICAL_SOURCE_FEATURES: List[str] = [
    "month",
    "day_of_year"
]

# Forecast Source Columns
FORECAST_COLUMNS: Dict[str, str] = {
    "NWP": "nwp_rainfall_mm",
    "AI": "ai_rainfall_mm",
    "ENSEMBLE": "ensemble_mean_rainfall_mm"
}

# Reference Target Column (Strictly prohibited from input features to prevent leakage)
REFERENCE_TARGET_COL: str = "reference_rainfall_mm"

# Softmax Temperature for Training Target Generation
SOFTMAX_TEMPERATURE: float = 0.5

# High-Performance XGBoost Model Hyperparameters (Optimized for 500k+ Records)
XGBOOST_PARAMS: Dict[str, Any] = {
    "n_estimators": 250,
    "max_depth": 6,
    "learning_rate": 0.04,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": 42,
    "n_jobs": -1
}

RANDOM_SEED: int = 42
