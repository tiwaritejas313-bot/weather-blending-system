"""
Inference API Module for SIH26081 — Hybrid Weather Forecast Blending System.

Exposes predict_blended_forecast(input_data) for single or batch forecast payloads.
Loads serialized models and scalers to compute dynamic weights, blended forecasts, and uncertainty metrics.
"""

import os
import sys

# Ensure project root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np

from ml import config
from ml.preprocessing import WeatherPreprocessor
from ml.weight_model import AdaptiveWeightModel
from ml.blender import ForecastBlender



_model_cache: Dict[str, Any] = {}


def _get_model_and_preprocessor():
    """Loads and caches the model and preprocessor singleton instances."""
    if "model" not in _model_cache:
        _model_cache["model"] = AdaptiveWeightModel.load(config.MODEL_PKL_PATH)
        preprocessor = WeatherPreprocessor()
        preprocessor.load(config.SCALER_PKL_PATH)
        _model_cache["preprocessor"] = preprocessor
    return _model_cache["model"], _model_cache["preprocessor"]


def predict_blended_forecast(
    input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Main prediction entry point for external applications and backend servers.
    
    Args:
        input_data: Dictionary, list of dictionaries, or pandas DataFrame containing forecast input parameters.
        
    Returns:
        Structured prediction result dictionary or list of dictionaries containing dynamic weights and blended forecast.
    """
    model, preprocessor = _get_model_and_preprocessor()

    is_single = isinstance(input_data, dict)
    if is_single:
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("input_data must be a dict, list of dicts, or pandas DataFrame")

    # Transform features using trained preprocessor
    X = preprocessor.transform(df)

    # Predict dynamic weights
    weights = model.predict_weights(X)

    results = []
    for i in range(len(df)):
        row = df.iloc[i]
        w_nwp, w_ai, w_ens = weights[i]

        nwp_f = float(row.get("nwp_rainfall_mm", 0.0))
        ai_f = float(row.get("ai_rainfall_mm", 0.0))
        ens_f = float(row.get("ensemble_mean_rainfall_mm", 0.0))
        ens_std = float(row.get("ensemble_std_rainfall_mm", 1.0))

        res = ForecastBlender.blend_single_observation(
            w_nwp=w_nwp,
            w_ai=w_ai,
            w_ens=w_ens,
            nwp_forecast=nwp_f,
            ai_forecast=ai_f,
            ensemble_forecast=ens_f,
            ensemble_std=ens_std,
        )
        res["weather_regime"] = row.get("weather_regime", "Unknown")
        res["lead_time_hours"] = int(row.get("lead_time_hours", 24))
        results.append(res)

    return results[0] if is_single else results


if __name__ == "__main__":
    # Test sample inference payload
    sample_input = {
        "latitude": 19.0760,
        "longitude": 72.8777,
        "month": 7,
        "day_of_year": 195,
        "lead_time_hours": 12,
        "temperature_2m_c": 27.5,
        "humidity_pct": 88.0,
        "surface_pressure_hpa": 1002.5,
        "u_wind_10m_ms": 4.2,
        "v_wind_10m_ms": 5.1,
        "weather_regime": "Monsoon",
        "nwp_rainfall_mm": 18.5,
        "ai_rainfall_mm": 21.2,
        "ensemble_mean_rainfall_mm": 19.8,
        "ensemble_std_rainfall_mm": 2.1,
        "nwp_historical_rmse": 3.8,
        "ai_historical_rmse": 2.2,
        "ensemble_historical_rmse": 3.1
    }

    print("🔍 Testing Inference API with sample payload...")
    output = predict_blended_forecast(sample_input)
    import json
    print("\n" + json.dumps(output, indent=2))
