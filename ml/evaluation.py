"""
Evaluation and Visualization Module for SIH26081 — Hybrid Weather Forecast Blending System.

Computes metrics (MAE, RMSE, R², Pearson r), baseline comparison matrices, relative improvement percentages,
stratified regime/lead-time analyses, and generates all 11 required visualization artifacts.
"""

import os
import sys

# Ensure project root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from ml import config


# Style configuration for publication-grade plots
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Helvetica"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


class ForecastEvaluator:
    """Evaluates multi-model forecasts against reference observations and creates visualizations."""

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates MAE, RMSE, R2, and Pearson correlation coefficient."""
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))

        if len(np.unique(y_pred)) > 1 and len(np.unique(y_true)) > 1:
            corr = float(np.corrcoef(y_true, y_pred)[0, 1])
        else:
            corr = 0.0

        return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4), "Correlation": round(corr, 4)}

    def evaluate_all_models(self, test_df: pd.DataFrame, blended_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compares Adaptive XGBoost Blend against:
        1. NWP alone
        2. AI alone
        3. Ensemble alone
        4. Equal-Weighted Average
        5. Adaptive XGBoost
        """
        y_ref = test_df["reference_rainfall_mm"].values
        nwp = test_df["nwp_rainfall_mm"].values
        ai = test_df["ai_rainfall_mm"].values
        ens = test_df["ensemble_mean_rainfall_mm"].values
        eq_weight = (nwp + ai + ens) / 3.0
        adaptive = blended_df["blended_rainfall"].values

        models_data = {
            "NWP": nwp,
            "AI": ai,
            "Ensemble": ens,
            "Equal Weight": eq_weight,
            "Adaptive XGBoost": adaptive,
        }

        metrics_list = []
        for model_name, y_pred in models_data.items():
            m = self.compute_metrics(y_ref, y_pred)
            m["Model"] = model_name
            metrics_list.append(m)

        summary_df = pd.DataFrame(metrics_list)[["Model", "MAE", "RMSE", "R2", "Correlation"]]

        # Calculate Improvement percentages over baselines
        adaptive_rmse = summary_df.loc[summary_df["Model"] == "Adaptive XGBoost", "RMSE"].values[0]
        
        improvements = []
        for _, row in summary_df.iterrows():
            m_name = row["Model"]
            if m_name == "Adaptive XGBoost":
                imp = 0.0
            else:
                base_rmse = row["RMSE"]
                imp = ((base_rmse - adaptive_rmse) / base_rmse) * 100.0 if base_rmse > 0 else 0.0
            improvements.append({"Baseline Model": m_name, "Baseline RMSE": row["RMSE"], "Adaptive RMSE": adaptive_rmse, "Improvement (%)": round(imp, 2)})

        improvement_df = pd.DataFrame(improvements)

        return summary_df, improvement_df

    def evaluate_by_strata(self, test_df: pd.DataFrame, blended_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluates model performance stratified by Weather Regime and Lead Time."""
        eval_df = test_df.copy()
        eval_df["blended_rainfall"] = blended_df["blended_rainfall"].values
        eval_df["Equal Weight"] = (eval_df["nwp_rainfall_mm"] + eval_df["ai_rainfall_mm"] + eval_df["ensemble_mean_rainfall_mm"]) / 3.0

        # By Weather Regime
        regime_results = []
        for regime, group in eval_df.groupby("weather_regime"):
            y_ref = group["reference_rainfall_mm"].values
            for m_col, label in [("nwp_rainfall_mm", "NWP"), ("ai_rainfall_mm", "AI"), ("ensemble_mean_rainfall_mm", "Ensemble"), ("Equal Weight", "Equal Weight"), ("blended_rainfall", "Adaptive XGBoost")]:
                m = self.compute_metrics(y_ref, group[m_col].values)
                m["Weather_Regime"] = regime
                m["Model"] = label
                regime_results.append(m)

        df_regime = pd.DataFrame(regime_results)

        # By Lead Time
        lt_results = []
        for lt, group in eval_df.groupby("lead_time_hours"):
            y_ref = group["reference_rainfall_mm"].values
            for m_col, label in [("nwp_rainfall_mm", "NWP"), ("ai_rainfall_mm", "AI"), ("ensemble_mean_rainfall_mm", "Ensemble"), ("Equal Weight", "Equal Weight"), ("blended_rainfall", "Adaptive XGBoost")]:
                m = self.compute_metrics(y_ref, group[m_col].values)
                m["Lead_Time_Hours"] = lt
                m["Model"] = label
                lt_results.append(m)

        df_lt = pd.DataFrame(lt_results)

        return df_regime, df_lt

    def generate_all_plots(
        self,
        test_df: pd.DataFrame,
        blended_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        df_regime: pd.DataFrame,
        df_lt: pd.DataFrame,
        plots_dir: str = config.PLOTS_DIR,
    ) -> None:
        """Generates all 11 required visual plots and saves them into plots_dir."""
        os.makedirs(plots_dir, exist_ok=True)
        colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#1b9e77"]

        # 1. RMSE Comparison Bar Chart
        plt.figure(figsize=(9, 5))
        sns.barplot(data=summary_df, x="Model", y="RMSE", palette=colors)
        plt.title("Model Performance Comparison — RMSE (Lower is Better)", fontsize=13, fontweight="bold", pad=12)
        plt.ylabel("RMSE (mm)")
        plt.xticks(rotation=15)
        for i, row in summary_df.iterrows():
            plt.text(i, row["RMSE"] + 0.05, f"{row['RMSE']:.2f}", ha="center", fontweight="bold", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "rmse_comparison.png"), dpi=300)
        plt.close()

        # 2. MAE Comparison Bar Chart
        plt.figure(figsize=(9, 5))
        sns.barplot(data=summary_df, x="Model", y="MAE", palette=colors)
        plt.title("Model Performance Comparison — MAE (Lower is Better)", fontsize=13, fontweight="bold", pad=12)
        plt.ylabel("MAE (mm)")
        plt.xticks(rotation=15)
        for i, row in summary_df.iterrows():
            plt.text(i, row["MAE"] + 0.05, f"{row['MAE']:.2f}", ha="center", fontweight="bold", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "mae_comparison.png"), dpi=300)
        plt.close()

        # Time series aggregated by date for weight trends
        ts_df = test_df.copy()
        ts_df["w_nwp"] = blended_df["nwp_weight"].values
        ts_df["w_ai"] = blended_df["ai_weight"].values
        ts_df["w_ens"] = blended_df["ensemble_weight"].values
        ts_daily = ts_df.groupby("date")[["w_nwp", "w_ai", "w_ens"]].mean().reset_index()

        # 3. Dynamic NWP Weight over Time
        plt.figure(figsize=(12, 4))
        plt.plot(ts_daily["date"], ts_daily["w_nwp"], color="#2b5c8f", linewidth=1.5)
        plt.title("Dynamic NWP Model Weight (w_NWP) Over Time", fontsize=12, fontweight="bold")
        plt.ylabel("NWP Weight")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "nwp_weight_time.png"), dpi=300)
        plt.close()

        # 4. Dynamic AI Weight over Time
        plt.figure(figsize=(12, 4))
        plt.plot(ts_daily["date"], ts_daily["w_ai"], color="#d95f02", linewidth=1.5)
        plt.title("Dynamic AI Model Weight (w_AI) Over Time", fontsize=12, fontweight="bold")
        plt.ylabel("AI Weight")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "ai_weight_time.png"), dpi=300)
        plt.close()

        # 5. Dynamic Ensemble Weight over Time
        plt.figure(figsize=(12, 4))
        plt.plot(ts_daily["date"], ts_daily["w_ens"], color="#7570b3", linewidth=1.5)
        plt.title("Dynamic Ensemble Model Weight (w_Ensemble) Over Time", fontsize=12, fontweight="bold")
        plt.ylabel("Ensemble Weight")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "ensemble_weight_time.png"), dpi=300)
        plt.close()

        # 6. Weight Distribution
        plt.figure(figsize=(9, 5))
        sns.kdeplot(ts_df["w_nwp"], label="NWP Weight", color="#2b5c8f", fill=True, alpha=0.3)
        sns.kdeplot(ts_df["w_ai"], label="AI Weight", color="#d95f02", fill=True, alpha=0.3)
        sns.kdeplot(ts_df["w_ens"], label="Ensemble Weight", color="#7570b3", fill=True, alpha=0.3)
        plt.title("Adaptive Weight Probability Distributions", fontsize=13, fontweight="bold")
        plt.xlabel("Weight Value")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "weight_distribution.png"), dpi=300)
        plt.close()

        # 7. Actual vs Blended Rainfall
        plt.figure(figsize=(8, 8))
        y_actual = test_df["reference_rainfall_mm"]
        y_blend = blended_df["blended_rainfall"]
        plt.scatter(y_actual, y_blend, alpha=0.3, color="#1b9e77", edgecolors="none")
        max_val = max(y_actual.max(), y_blend.max())
        plt.plot([0, max_val], [0, max_val], "k--", label="1:1 Perfect Forecast")
        plt.title("Reference Rainfall vs Adaptive Blended Forecast", fontsize=13, fontweight="bold")
        plt.xlabel("Actual Reference Rainfall (mm)")
        plt.ylabel("Adaptive Blended Rainfall (mm)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "actual_vs_blended.png"), dpi=300)
        plt.close()

        # 8. Error Comparison
        plt.figure(figsize=(10, 5))
        err_df = pd.DataFrame({
            "NWP": np.abs(test_df["nwp_rainfall_mm"] - y_actual),
            "AI": np.abs(test_df["ai_rainfall_mm"] - y_actual),
            "Ensemble": np.abs(test_df["ensemble_mean_rainfall_mm"] - y_actual),
            "Equal Weight": np.abs((test_df["nwp_rainfall_mm"] + test_df["ai_rainfall_mm"] + test_df["ensemble_mean_rainfall_mm"]) / 3.0 - y_actual),
            "Adaptive XGBoost": np.abs(y_blend - y_actual),
        })
        sns.boxplot(data=err_df, palette=colors, showfliers=False)
        plt.title("Absolute Forecast Error Distributions Across Models", fontsize=13, fontweight="bold")
        plt.ylabel("Absolute Error (mm)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "error_comparison.png"), dpi=300)
        plt.close()

        # 9. Performance by Lead Time
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df_lt, x="Lead_Time_Hours", y="RMSE", hue="Model", style="Model", markers=True, dashes=False, palette=colors, linewidth=2.0)
        plt.title("Forecast Performance (RMSE) Across Lead Times (6h - 120h)", fontsize=13, fontweight="bold")
        plt.xlabel("Forecast Lead Time (Hours)")
        plt.ylabel("RMSE (mm)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "performance_by_leadtime.png"), dpi=300)
        plt.close()

        # 10. Performance by Season / Weather Regime
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_regime, x="Weather_Regime", y="RMSE", hue="Model", palette=colors)
        plt.title("Forecast Performance (RMSE) Stratified by Weather Regime", fontsize=13, fontweight="bold")
        plt.xlabel("Weather Regime")
        plt.ylabel("RMSE (mm)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "performance_by_season.png"), dpi=300)
        plt.close()

        # 11. Geographic Weight Map
        plt.figure(figsize=(9, 6))
        geo_df = ts_df.groupby(["latitude", "longitude"])[["w_nwp", "w_ai", "w_ens"]].mean().reset_index()
        plt.scatter(geo_df["longitude"], geo_df["latitude"], c=geo_df["w_ai"], cmap="plasma", s=250, edgecolors="black", linewidth=1.5)
        plt.colorbar(label="Mean AI Weight (w_AI)")
        for _, row in geo_df.iterrows():
            plt.annotate(f"w_AI={row['w_ai']:.2f}", (row["longitude"] + 0.3, row["latitude"] + 0.3), fontsize=9, fontweight="bold")
        plt.title("Geographic AI Weight Distribution across Indian Stations", fontsize=13, fontweight="bold")
        plt.xlabel("Longitude (°E)")
        plt.ylabel("Latitude (°N)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "geographic_weight_map.png"), dpi=300)
        plt.close()

        print(f"📊 Successfully generated all 11 plots saved in: '{plots_dir}'")
