"""
Training Pipeline Orchestrator for SIH26081 — Hybrid Weather Forecast Blending System.

Loads dataset, performs chronological train/val/test splitting, trains Adaptive XGBoost Weight Model,
evaluates against all baselines, generates plots, and serializes trained models.
"""

import os
import sys

# Ensure project root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pandas as pd
import numpy as np

from ml import config
from ml.data_loader import WeatherDataLoader
from ml.preprocessing import WeatherPreprocessor
from ml.weight_model import AdaptiveWeightModel
from ml.blender import ForecastBlender
from ml.evaluation import ForecastEvaluator



def run_training_pipeline() -> None:
    print("=" * 70)
    print(" 🚀 SIH26081 — HYBRID AI-NWP MULTI-MODEL FORECAST BLENDING SYSTEM")
    print("=" * 70)

    # Step 1: Load Data
    print("\n[Step 1/7] Loading Meteorological Dataset...")
    loader = WeatherDataLoader()
    df = loader.load_data()

    # Step 2: Chronological Split
    print("\n[Step 2/7] Executing Chronological Split (No Data Leakage)...")
    df_train, df_val, df_test = loader.get_chronological_splits(df)

    # Step 3: Feature Engineering & Preprocessing
    print("\n[Step 3/7] Preprocessing Features & Fitting StandardScaler...")
    preprocessor = WeatherPreprocessor()
    X_train, feature_names = preprocessor.fit_transform(df_train)
    X_val = preprocessor.transform(df_val)
    X_test = preprocessor.transform(df_test)

    print(f"   • Engineered Input Features ({len(feature_names)}): {feature_names}")

    # Step 4: Train Adaptive Weight Learning Model
    print("\n[Step 4/7] Training XGBoost Multi-Output Softmax Weight Model...")
    weight_model = AdaptiveWeightModel()
    weight_model.fit(X_train, df_train)

    # Step 5: Inference & Blending on Test Set (Holdout 2024 Data)
    print("\n[Step 5/7] Blending Multi-Model Forecasts for Holdout Test Set (2024)...")
    test_weights = weight_model.predict_weights(X_test)
    
    blended_test = ForecastBlender.blend_forecasts(
        weights=test_weights,
        nwp_rain=df_test["nwp_rainfall_mm"].values,
        ai_rain=df_test["ai_rainfall_mm"].values,
        ensemble_rain=df_test["ensemble_mean_rainfall_mm"].values,
        ensemble_std=df_test["ensemble_std_rainfall_mm"].values,
    )

    # Verify Dynamic Weights Properties
    weight_sums = test_weights.sum(axis=1)
    assert np.allclose(weight_sums, 1.0), "Error: Dynamic weights do not sum to 1.0!"
    assert np.all((test_weights >= 0.0) & (test_weights <= 1.0)), "Error: Weights out of bounds [0, 1]!"
    print("   ✅ Dynamic weights verified: 0 <= w_i <= 1 and sum(weights) = 1.0 across all samples.")

    # Step 6: Comprehensive Evaluation & Baseline Comparisons
    print("\n[Step 6/7] Evaluating Baselines vs. Adaptive XGBoost Blend...")
    evaluator = ForecastEvaluator()
    summary_df, improvement_df = evaluator.evaluate_all_models(df_test, blended_test)
    df_regime, df_lt = evaluator.evaluate_by_strata(df_test, blended_test)

    print("\n" + "=" * 55)
    print(" 🏆 MODEL PERFORMANCE COMPARISON MATRIX (HOLDOUT TEST SET 2024)")
    print("=" * 55)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 65)
    print(" 📈 ADAPTIVE XGBOOST IMPROVEMENT OVER BASELINES (%)")
    print("=" * 65)
    print(improvement_df.to_string(index=False))

    # Generate Visualizations
    print("\nGenerating all 11 plot artifacts in 'results/plots/'...")
    evaluator.generate_all_plots(df_test, blended_test, summary_df, df_regime, df_lt)

    # Step 7: Model & Config Serialization
    print("\n[Step 7/7] Serializing Model Artifacts & Configuration...")
    weight_model.save(config.MODEL_PKL_PATH)
    preprocessor.save(config.SCALER_PKL_PATH)

    config_payload = {
        "train_end_date": config.TRAIN_END_DATE,
        "val_end_date": config.VAL_END_DATE,
        "test_start_date": config.TEST_START_DATE,
        "feature_names": feature_names,
        "softmax_temperature": config.SOFTMAX_TEMPERATURE,
        "xgboost_params": config.XGBOOST_PARAMS,
        "random_seed": config.RANDOM_SEED,
        "test_metrics": summary_df.to_dict(orient="records"),
        "improvements": improvement_df.to_dict(orient="records"),
    }

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(config.CONFIG_JSON_PATH, "w") as f:
        json.dump(config_payload, f, indent=2)

    print(f"💾 Saved configuration manifest to '{config.CONFIG_JSON_PATH}'")
    print("\n🎉 SIH26081 Training & Evaluation Pipeline Completed Successfully!\n")


if __name__ == "__main__":
    run_training_pipeline()
