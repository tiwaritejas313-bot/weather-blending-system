"""
Weight Learning Model Module for SIH26081 — Hybrid Weather Forecast Blending System.

Implements the Adaptive Weight Learning architecture using XGBoost multi-output regression
followed by a Softmax normalization layer to output dynamically calibrated weights:
    w_NWP, w_AI, w_Ensemble s.t. sum(weights) = 1 and 0 <= weight <= 1.
"""

import os
import sys

# Ensure project root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import joblib

from ml import config



def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Computes numerically stable softmax across the specified axis."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def compute_target_weights(
    nwp_rain: np.ndarray,
    ai_rain: np.ndarray,
    ensemble_rain: np.ndarray,
    reference_rain: np.ndarray,
    temperature: float = config.SOFTMAX_TEMPERATURE
) -> np.ndarray:
    """
    Computes target reliability weights using absolute forecast errors.
    Lower historical/current forecast error yields higher target weight importance.
    
    Target formula:
        y_target_i = exp(-temperature * Error_i) / sum(exp(-temperature * Error_j))
    """
    err_nwp = np.abs(nwp_rain - reference_rain)
    err_ai = np.abs(ai_rain - reference_rain)
    err_ens = np.abs(ensemble_rain - reference_rain)

    errors = np.column_stack([err_nwp, err_ai, err_ens])
    
    # Softmax over negative scaled errors (lower error = higher target probability)
    target_weights = softmax(-temperature * errors, axis=1)
    return target_weights


class AdaptiveWeightModel:
    """
    XGBoost Multi-Output Regressor + Softmax Normalization for Adaptive Weather Forecast Blending.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or config.XGBOOST_PARAMS
        base_xgb = xgb.XGBRegressor(**self.params)
        self.model = MultiOutputRegressor(base_xgb)
        self.is_fitted = False

    def fit(self, X: np.ndarray, df_train: pd.DataFrame) -> "AdaptiveWeightModel":
        """
        Trains the XGBoost regressor on input features X to predict target softmax weight distributions.
        
        Args:
            X: Standardized feature matrix (must NOT contain reference_rainfall_mm).
            df_train: Raw train DataFrame containing model forecasts and reference observation.
        """
        nwp_rain = df_train["nwp_rainfall_mm"].values
        ai_rain = df_train["ai_rainfall_mm"].values
        ens_rain = df_train["ensemble_mean_rainfall_mm"].values
        ref_rain = df_train["reference_rainfall_mm"].values

        # Compute ground truth target reliability weights
        Y_target = compute_target_weights(nwp_rain, ai_rain, ens_rain, ref_rain)

        # Train Multi-Output XGBoost model
        self.model.fit(X, Y_target)
        self.is_fitted = True
        return self

    def predict_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts raw model scores and applies Softmax normalization layer.
        
        Returns:
            Numpy array of shape (N, 3) where columns are [w_NWP, w_AI, w_Ensemble].
            Guarantees 0 <= weight <= 1 and sum(weights) = 1.0.
        """
        if not self.is_fitted:
            raise RuntimeError("AdaptiveWeightModel is not fitted yet!")

        raw_scores = self.model.predict(X)
        weights = softmax(raw_scores, axis=1)
        return weights

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Extracts gain-based feature importances across each output weight model.
        Returns dictionary mapping output model (NWP, AI, Ensemble) to feature importance scores.
        """
        if not self.is_fitted:
            return {}

        importances = {}
        target_names = ["w_NWP", "w_AI", "w_Ensemble"]
        
        for idx, estimator in enumerate(self.model.estimators_):
            gains = estimator.feature_importances_
            feat_imp = {feat: float(np.round(gain, 4)) for feat, gain in zip(feature_names, gains)}
            # Sort descending
            sorted_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
            importances[target_names[idx]] = sorted_imp

        return importances

    def save(self, path: str = config.MODEL_PKL_PATH) -> None:

        """Saves model instance using joblib."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"💾 Adaptive Weight Model saved to '{path}'")

    @classmethod
    def load(cls, path: str = config.MODEL_PKL_PATH) -> "AdaptiveWeightModel":
        """Loads model instance from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at '{path}'")
        return joblib.load(path)
