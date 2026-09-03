# SIH26081 — Hybrid AI-NWP Multi-Model Forecast Blending System

> **Smart India Hackathon 2026 (SIH26081)**  
> **Problem Statement**: Build a dynamic, machine-learning-based hybrid weather forecasting system that intelligently combines multiple weather forecast models (NWP, AI, and Ensemble) based on location, season, lead time, atmospheric conditions, forecast uncertainty, and historical performance.

---

## 🌟 Key Highlights & Scientific Principles

- **Adaptive Weight Learning vs. Direct Rainfall Prediction**: Instead of directly predicting rainfall, the system uses an XGBoost Multi-Output Regressor followed by a **Softmax Normalization Layer** to learn adaptive confidence weights ($w_{\text{NWP}}, w_{\text{AI}}, w_{\text{Ensemble}}$).
- **Strict Mathematical Bounds**: Guarantees that $0 \le w_i \le 1$ and $w_{\text{NWP}} + w_{\text{AI}} + w_{\text{Ensemble}} = 1.0$ for every forecast instance.
- **Zero Target Leakage Guarantee**: Ground truth `reference_rainfall_mm` is strictly isolated. It is used **only** during training to construct target Softmax reliability distributions and during post-prediction evaluation. It is **never** included as an input feature.
- **Strict Chronological Splitting**: Weather records are split chronologically (Train: 2019–2022, Validation: 2023, Holdout Test: 2024) to ensure no future temporal information corrupts training.
- **Rigorous Baseline Benchmarking**: Compares the adaptive blend against 5 baselines:
  1. NWP alone
  2. AI alone
  3. Ensemble alone
  4. Equal-Weighted Average ($0.3333 \times NWP + 0.3333 \times AI + 0.3333 \times Ensemble$)
  5. **Adaptive XGBoost Blend**
- **Real-Data Compatibility**: Built modularly so that the initial synthetic CSV dataset (`SIH26081_ML_training_sample.csv`) can be replaced with real meteorological datasets from **NCMRWF IMDAA**, **ERA5**, **GEFS**, and **GraphCast** without changing the architecture.

---

## 📐 Mathematical Formulation

### 1. Adaptive Weight Normalization Layer
$$\begin{pmatrix} s_{\text{NWP}} \\ s_{\text{AI}} \\ s_{\text{Ensemble}} \end{pmatrix} = \text{XGBoost}(X)$$

$$w_i = \frac{\exp(s_i)}{\sum_{j \in \{\text{NWP, AI, Ensemble}\}} \exp(s_j)}$$

### 2. Blended Forecast Equation
$$\text{Final\_Forecast} = w_{\text{NWP}} \cdot \text{NWP\_Forecast} + w_{\text{AI}} \cdot \text{AI\_Forecast} + w_{\text{Ensemble}} \cdot \text{Ensemble\_Forecast}$$

### 3. Training Target Reliability Generation
For training sample $k$, error relative to `reference_rainfall_mm` is:
$$E_i = |\text{Forecast}_i - \text{Reference\_Rainfall}|$$

Target Softmax distribution:
$$y_{\text{target}, i} = \frac{\exp(-\tau \cdot E_i)}{\sum_j \exp(-\tau \cdot E_j)}$$
where $\tau = 0.5$ is the scaling temperature parameter.

---

## 📁 Repository Structure

```
sih26081-hybrid-forecast/
├── data/
│   └── SIH26081_ML_training_sample.csv  # Synthetic meteorological sample data
├── ml/
│   ├── __init__.py
│   ├── config.py             # Feature definitions, paths, split thresholds
│   ├── data_loader.py        # Chronological dataset split & schema verification
│   ├── preprocessing.py     # Cyclical encodings, scaler, zero-leakage check
│   ├── weight_model.py       # XGBoost Multi-Output Regressor + Softmax layer
│   ├── blender.py            # Blending engine & uncertainty quantifier
│   ├── evaluation.py         # Baseline metric comparison & 11 plot generators
│   ├── train.py              # Full end-to-end training pipeline runner
│   └── predict.py            # Standalone inference API module
├── models/
│   ├── blending_model.pkl    # Serialized XGBoost Weight Model
│   ├── scaler.pkl            # Serialized StandardScaler & feature schema
│   └── config.json           # Model manifest & evaluation summary metrics
├── results/
│   └── plots/                # 11 Generated visualization artifacts
├── api_server.py             # FastAPI REST Server for backend integration (SIH_ISOBAR)
├── generate_sample_data.py   # Meteorological sample data generator
├── requirements.txt          # Dependencies list
└── README.md                 # System documentation
```

---

## 🚀 Quickstart & Execution Guide

### 1. Installation & Environment Setup
```bash
# Navigate to the workspace directory
cd /Users/tejastiwari/.gemini/antigravity/scratch/sih26081-hybrid-forecast

# Install required dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Meteorological Dataset (2019-2024)
```bash
python generate_sample_data.py
```

### 3. Train & Evaluate the Hybrid Blending Model
```bash
python ml/train.py
```
*Outputs printed to terminal include chronological split stats, multi-model evaluation metrics (MAE, RMSE, $R^2$, Correlation), percentage improvements over baselines, and 11 generated plots saved to `results/plots/`.*

### 4. Run Single/Batch Forecast Predictions
```bash
python ml/predict.py
```

### 5. Launch REST API Microservice for Backend Integration (`SIH_ISOBAR`)
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```
- Open Swagger Docs: `http://localhost:8000/docs`
- Health check: `GET http://localhost:8000/health`
- Predict endpoint: `POST http://localhost:8000/predict`

---

## 📊 Visualizations Generated (`results/plots/`)

1. `rmse_comparison.png`: Bar chart comparing RMSE across all 5 model configurations.
2. `mae_comparison.png`: Bar chart comparing MAE across all 5 model configurations.
3. `nwp_weight_time.png`: Dynamic $w_{\text{NWP}}$ allocation over time series.
4. `ai_weight_time.png`: Dynamic $w_{\text{AI}}$ allocation over time series.
5. `ensemble_weight_time.png`: Dynamic $w_{\text{Ensemble}}$ allocation over time series.
6. `weight_distribution.png`: KDE distribution curves of dynamic weights.
7. `actual_vs_blended.png`: Scatter plot comparing reference rainfall against blended forecasts.
8. `error_comparison.png`: Box plot showing absolute error distributions.
9. `performance_by_leadtime.png`: Line chart tracking RMSE across lead times (6h, 12h, 24h, 48h, 72h, 120h).
10. `performance_by_season.png`: Grouped bar chart comparing performance across weather regimes (Monsoon, Winter, Summer, Transition).
11. `geographic_weight_map.png`: Spatial heatmap of dynamic weight allocation across geographical coordinates.

---

## 🔌 Integration with Backend (`SIH_ISOBAR`)

This ML microservice easily connects with your backend repository (`https://github.com/abhixw/SIH_ISOBAR`):

### Option A: Python Module Import
```python
from ml.predict import predict_blended_forecast

input_payload = {
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

result = predict_blended_forecast(input_payload)
print(result)
```

### Option B: HTTP REST Call (Node.js / Express or Python Backend)
```javascript
const axios = require('axios');

async function getBlendedForecast(forecastData) {
    const response = await axios.post('http://localhost:8000/predict', forecastData);
    return response.data;
}
```

---

## 🌍 Migrating to Real Meteorological Data (ERA5 / IMDAA / GEFS / GraphCast)

To swap the synthetic sample dataset with real NetCDF/GRIB2 meteorological observations:
1. Extract ERA5 or IMDAA reanalysis rainfall to `reference_rainfall_mm`.
2. Extract NCMRWF/GEFS predictions to `nwp_rainfall_mm`, `ensemble_mean_rainfall_mm`, `ensemble_std_rainfall_mm`.
3. Extract GraphCast / FourCastNet forecasts to `ai_rainfall_mm`.
4. Run `python ml/train.py`. The architecture, scaling, Softmax normalization, and evaluation suite remain identical!
