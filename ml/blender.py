"""
Forecast Blender Engine Module for SIH26081 — Hybrid Weather Forecast Blending System.

Applies predicted dynamic weights to combine NWP, AI, and Ensemble forecasts,
calculates blended forecast uncertainty metrics, IMD Severe Weather Warning Advisories,
and Model Reliability Scores.
"""

from typing import Dict, Any, Union, List
import numpy as np
import pandas as pd


def get_imd_warning_level(rainfall_mm: float) -> Dict[str, str]:
    """
    Categorizes rainfall intensity according to IMD (India Meteorological Department) standards:
    - Green: Light Rain (< 15.5 mm/day)
    - Yellow: Moderate Rain / Be Updated (15.5 - 64.4 mm/day)
    - Orange: Heavy Rain / Be Prepared (64.5 - 115.5 mm/day)
    - Red: Extremely Heavy Rain / Take Action (> 115.5 mm/day)
    """
    if rainfall_mm >= 115.5:
        return {
            "level": "RED",
            "color": "#ef4444",
            "category": "Extremely Heavy Rainfall",
            "advisory": "TAKE ACTION: Flash flood threat & extreme localized inundation imminent."
        }
    elif rainfall_mm >= 64.5:
        return {
            "level": "ORANGE",
            "color": "#f97316",
            "category": "Heavy to Very Heavy Rainfall",
            "advisory": "BE PREPARED: Heavy rainfall may cause waterlogging and urban flooding."
        }
    elif rainfall_mm >= 15.5:
        return {
            "level": "YELLOW",
            "color": "#eab308",
            "category": "Moderate Rainfall",
            "advisory": "BE UPDATED: Moderate rain expected. Keep monitoring weather updates."
        }
    else:
        return {
            "level": "GREEN",
            "color": "#10b981",
            "category": "Light / Nil Rainfall",
            "advisory": "NO WARNING: Normal atmospheric conditions expected."
        }


class ForecastBlender:
    """Combines multi-model weather forecasts using adaptive weights."""

    @staticmethod
    def blend_forecasts(
        weights: np.ndarray,
        nwp_rain: np.ndarray,
        ai_rain: np.ndarray,
        ensemble_rain: np.ndarray,
        ensemble_std: np.ndarray,
    ) -> pd.DataFrame:
        """Calculates blended rainfall forecast and stores component weights."""
        w_nwp = weights[:, 0]
        w_ai = weights[:, 1]
        w_ens = weights[:, 2]

        blended_rainfall = (
            w_nwp * nwp_rain
            + w_ai * ai_rain
            + w_ens * ensemble_rain
        )
        
        blended_rainfall = np.clip(blended_rainfall, 0.0, None)

        disagreement = np.sqrt(
            w_nwp * np.square(nwp_rain - blended_rainfall)
            + w_ai * np.square(ai_rain - blended_rainfall)
            + w_ens * np.square(ensemble_rain - blended_rainfall)
        )

        blended_uncertainty = np.sqrt(np.square(disagreement) + w_ens * np.square(ensemble_std))

        res_df = pd.DataFrame({
            "nwp_weight": np.round(w_nwp, 4),
            "ai_weight": np.round(w_ai, 4),
            "ensemble_weight": np.round(w_ens, 4),
            "blended_rainfall": np.round(blended_rainfall, 2),
            "blended_uncertainty": np.round(blended_uncertainty, 2),
        })

        return res_df

    @staticmethod
    def blend_single_observation(
        w_nwp: float,
        w_ai: float,
        w_ens: float,
        nwp_forecast: float,
        ai_forecast: float,
        ensemble_forecast: float,
        ensemble_std: float = 1.0,
    ) -> Dict[str, Any]:
        """Calculates single-sample forecast blend, warning advisories, and model reliability."""
        blended = w_nwp * nwp_forecast + w_ai * ai_forecast + w_ens * ensemble_forecast
        blended = max(0.0, float(blended))

        disagreement = np.sqrt(
            w_nwp * (nwp_forecast - blended) ** 2
            + w_ai * (ai_forecast - blended) ** 2
            + w_ens * (ensemble_forecast - blended) ** 2
        )
        uncertainty = float(np.sqrt(disagreement ** 2 + w_ens * (ensemble_std ** 2)))

        # Calibrated 80% and 95% Prediction Bounds
        margin_80 = 1.28 * uncertainty
        margin_95 = 1.96 * uncertainty

        ci_80_low = max(0.0, round(blended - margin_80, 2))
        ci_80_high = round(blended + margin_80, 2)
        ci_95_low = max(0.0, round(blended - margin_95, 2))
        ci_95_high = round(blended + margin_95, 2)

        # IMD Severe Warning Categorization
        warning_info = get_imd_warning_level(blended)

        # Model Reliability Scores (relative confidence percentage)
        max_weight = max(w_nwp, w_ai, w_ens)
        reliability_scores = {
            "nwp_reliability": round(float((w_nwp / max_weight) * 100.0), 1),
            "ai_reliability": round(float((w_ai / max_weight) * 100.0), 1),
            "ensemble_reliability": round(float((w_ens / max_weight) * 100.0), 1),
        }

        return {
            "nwp_weight": round(float(w_nwp), 4),
            "ai_weight": round(float(w_ai), 4),
            "ensemble_weight": round(float(w_ens), 4),
            "nwp_forecast": round(float(nwp_forecast), 2),
            "ai_forecast": round(float(ai_forecast), 2),
            "ensemble_forecast": round(float(ensemble_forecast), 2),
            "blended_forecast": round(blended, 2),
            "blended_uncertainty": round(uncertainty, 2),
            "confidence_bounds_80": [ci_80_low, ci_80_high],
            "confidence_bounds_95": [ci_95_low, ci_95_high],
            "warning": warning_info,
            "reliability_scores": reliability_scores
        }
