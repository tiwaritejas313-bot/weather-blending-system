"""
Vercel Serverless Entrypoint for SIH26081 Weather Forecast Blending System.
"""

from typing import List, Dict, Any, Union, Optional
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import json
import os
import sys
import pandas as pd
import io

# Ensure root directory is in sys.path for ml package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml import config
from ml.predict import predict_blended_forecast, _get_model_and_preprocessor

app = FastAPI(
    title="SIH26081 — Operational AI-NWP Weather Forecast Blending Engine",
    description="Institutional Multi-Model Dynamic Weighting & Hydro-Meteorological Blending Microservice",
    version="2.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastInputSchema(BaseModel):
    latitude: float = Field(..., example=19.0760, description="Latitude in degrees N")
    longitude: float = Field(..., example=72.8777, description="Longitude in degrees E")
    month: int = Field(..., ge=1, le=12, example=7, description="Month of year (1-12)")
    day_of_year: int = Field(..., ge=1, le=366, example=195, description="Day of year (1-366)")
    lead_time_hours: int = Field(..., example=12, description="Forecast lead time in hours (6, 12, 24, 48, 72, 120)")
    temperature_2m_c: float = Field(..., example=27.5, description="2m Temperature in °C")
    humidity_pct: float = Field(..., example=88.0, description="Relative Humidity percentage")
    surface_pressure_hpa: float = Field(..., example=1002.5, description="Surface Pressure in hPa")
    u_wind_10m_ms: float = Field(..., example=4.2, description="U Wind vector component at 10m (m/s)")
    v_wind_10m_ms: float = Field(..., example=5.1, description="V Wind vector component at 10m (m/s)")
    weather_regime: str = Field("Monsoon", example="Monsoon", description="Regime: Monsoon, Winter, Summer, Transition")
    nwp_rainfall_mm: float = Field(..., example=18.5, description="NWP Forecast Rainfall in mm")
    ai_rainfall_mm: float = Field(..., example=21.2, description="AI Forecast Rainfall in mm")
    ensemble_mean_rainfall_mm: float = Field(..., example=19.8, description="Ensemble Mean Forecast Rainfall in mm")
    ensemble_std_rainfall_mm: float = Field(1.0, example=2.1, description="Ensemble Standard Deviation (Uncertainty)")
    nwp_historical_rmse: float = Field(3.5, example=3.8, description="Historical NWP RMSE")
    ai_historical_rmse: float = Field(2.5, example=2.2, description="Historical AI RMSE")
    ensemble_historical_rmse: float = Field(3.0, example=3.1, description="Historical Ensemble RMSE")


class WarningSchema(BaseModel):
    level: str
    color: str
    category: str
    advisory: str


class ReliabilitySchema(BaseModel):
    nwp_reliability: float
    ai_reliability: float
    ensemble_reliability: float


class BlendedForecastResponseSchema(BaseModel):
    nwp_weight: float
    ai_weight: float
    ensemble_weight: float
    nwp_forecast: float
    ai_forecast: float
    ensemble_forecast: float
    blended_forecast: float
    blended_uncertainty: float
    confidence_bounds_80: List[float]
    confidence_bounds_95: List[float]
    warning: Optional[WarningSchema] = None
    reliability_scores: Optional[ReliabilitySchema] = None
    weather_regime: str
    lead_time_hours: int


STATIONS_DATABASE = [
    {"id": "mumbai", "name": "Mumbai Met Centre", "zone": "West Coast / Konkan", "lat": 19.0760, "lon": 72.8777, "elev_m": 14, "nwp": 42.5, "ai": 48.2, "ens": 45.0, "temp": 28.5, "humidity": 89.0, "pressure": 1004.2, "regime": "Monsoon"},
    {"id": "delhi", "name": "Delhi Regional MC", "zone": "North Plains / NCR", "lat": 28.6139, "lon": 77.2090, "elev_m": 216, "nwp": 12.0, "ai": 10.5, "ens": 11.2, "temp": 34.0, "humidity": 55.0, "pressure": 1008.5, "regime": "Summer"},
    {"id": "chennai", "name": "Chennai Cyclone Warning Ctr", "zone": "South East Coast", "lat": 13.0827, "lon": 80.2707, "elev_m": 7, "nwp": 28.5, "ai": 32.0, "ens": 30.1, "temp": 30.2, "humidity": 82.0, "pressure": 1006.1, "regime": "Monsoon"},
    {"id": "kolkata", "name": "Kolkata Alipore Met Office", "zone": "Gangetic West Bengal", "lat": 22.5726, "lon": 88.3639, "elev_m": 9, "nwp": 35.0, "ai": 38.5, "ens": 36.2, "temp": 29.0, "humidity": 85.0, "pressure": 1003.8, "regime": "Monsoon"},
    {"id": "cherrapunji", "name": "Sohra / Cherrapunji Observatory", "zone": "Northeast Plateau (High Precipitation)", "lat": 25.2986, "lon": 91.7308, "elev_m": 1484, "nwp": 112.0, "ai": 125.0, "ens": 118.5, "temp": 22.0, "humidity": 98.0, "pressure": 995.4, "regime": "Monsoon"},
    {"id": "bengaluru", "name": "Bengaluru Met Centre", "zone": "South Interior Karnataka", "lat": 12.9716, "lon": 77.5946, "elev_m": 920, "nwp": 14.2, "ai": 16.0, "ens": 15.1, "temp": 24.5, "humidity": 72.0, "pressure": 1012.0, "regime": "Transition"},
    {"id": "srinagar", "name": "Srinagar Weather Station", "zone": "Western Himalayas", "lat": 34.0837, "lon": 74.7973, "elev_m": 1585, "nwp": 8.5, "ai": 7.2, "ens": 7.9, "temp": 18.0, "humidity": 60.0, "pressure": 1015.2, "regime": "Winter"},
    {"id": "bhubaneswar", "name": "Bhubaneswar Cyclone Office", "zone": "Odisha Coastal Zone", "lat": 20.2961, "lon": 85.8245, "elev_m": 45, "nwp": 52.0, "ai": 58.0, "ens": 54.5, "temp": 29.5, "humidity": 91.0, "pressure": 1001.5, "regime": "Monsoon"},
]



@app.get("/")
def serve_dashboard():
    static_file = os.path.join(config.BASE_DIR, "static", "index.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {"message": "SIH26081 Blending System API Online. Dashboard HTML not found."}


@app.get("/about")
def serve_landing():
    static_file = os.path.join(config.BASE_DIR, "static", "landing.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {"message": "Landing page HTML not found."}


@app.get("/classic")
def serve_classic_dashboard():
    static_file = os.path.join(config.BASE_DIR, "static", "index-classic.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {"message": "Classic dashboard HTML not found."}


@app.get("/modern")
def serve_modern_dashboard():
    static_file = os.path.join(config.BASE_DIR, "static", "index-modern.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {"message": "Modern dashboard HTML not found."}


@app.get("/health")
def health_check():
    return {"status": "operational", "system": "SIH26081 Multi-Model Blending Microservice", "version": "2.5.0"}


@app.get("/stations")
def get_stations():
    return STATIONS_DATABASE


@app.get("/metrics")
def get_model_metrics():
    if not os.path.exists(config.CONFIG_JSON_PATH):
        raise HTTPException(status_code=404, detail="Model configuration manifest not found.")
    with open(config.CONFIG_JSON_PATH, "r") as f:
        data = json.load(f)
    return {
        "test_metrics": data.get("test_metrics", []),
        "improvements": data.get("improvements", []),
        "features": data.get("feature_names", [])
    }


@app.get("/diagnostics")
def get_model_diagnostics():
    try:
        model, preprocessor = _get_model_and_preprocessor()
        importances = model.get_feature_importances(preprocessor.feature_names)
        
        with open(config.CONFIG_JSON_PATH, "r") as f:
            manifest = json.load(f)

        return {
            "feature_importances": importances,
            "test_summary": manifest.get("test_metrics", []),
            "improvements": manifest.get("improvements", []),
            "xgboost_parameters": manifest.get("xgboost_params", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=Union[BlendedForecastResponseSchema, List[BlendedForecastResponseSchema]])
def predict_endpoint(payload: Union[ForecastInputSchema, List[ForecastInputSchema]]):
    try:
        if isinstance(payload, list):
            input_dicts = [item.dict() for item in payload]
        else:
            input_dicts = payload.dict()
        
        result = predict_blended_forecast(input_dicts)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/export/report")
def export_bulletin_report(station: str = "mumbai", lead_time: int = 12):
    """Generates an official downloadable CSV forecast bulletin report."""
    st = next((s for s in STATIONS_DATABASE if s["id"] == station), STATIONS_DATABASE[0])
    
    sample_input = {
        "latitude": st["lat"],
        "longitude": st["lon"],
        "month": 7,
        "day_of_year": 195,
        "lead_time_hours": lead_time,
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
        "nwp_historical_rmse": 3.5,
        "ai_historical_rmse": 2.5,
        "ensemble_historical_rmse": 3.0
    }

    res = predict_blended_forecast(sample_input)
    
    report_data = [{
        "Station_Name": st["name"],
        "Zone": st["zone"],
        "Latitude": st["lat"],
        "Longitude": st["lon"],
        "Lead_Time_Hours": lead_time,
        "Weather_Regime": "Monsoon",
        "NWP_Forecast_mm": res["nwp_forecast"],
        "AI_Forecast_mm": res["ai_forecast"],
        "Ensemble_Forecast_mm": res["ensemble_forecast"],
        "w_NWP": res["nwp_weight"],
        "w_AI": res["ai_weight"],
        "w_Ensemble": res["ensemble_weight"],
        "Blended_Forecast_mm": res["blended_forecast"],
        "Blended_Uncertainty_mm": res["blended_uncertainty"],
        "CI_80_Low": res["confidence_bounds_80"][0],
        "CI_80_High": res["confidence_bounds_80"][1],
        "IMD_Warning_Level": res["warning"]["level"],
        "IMD_Advisory": res["warning"]["advisory"]
    }]

    df_rep = pd.DataFrame(report_data)
    stream = io.StringIO()
    df_rep.to_csv(stream, index=False)

    return Response(
        content=stream.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=IMD_Blended_Forecast_Bulletin_{st['id']}.csv"}
    )
