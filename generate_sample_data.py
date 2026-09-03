"""
Comprehensive 10-Year Meteorological Multi-Region Dataset Generator for SIH26081.

Generates ~550,000 multi-model weather forecast records spanning 2015-01-01 to 2024-12-31
across 25 key geographical grid locations representing ALL Indian Meteorological Subdivisions.
"""

import os
import numpy as np
import pandas as pd

def generate_sih_weather_dataset(output_path: str, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    # 10-Year Timeline: 2015-01-01 to 2024-12-31 (3,653 days)
    dates = pd.date_range(start="2015-01-01", end="2024-12-31", freq="D")

    # 25 Diverse Indian Geographical Locations across 8 Climate Zones
    locations = [
        # West Coast & Konkan
        {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "rain_mult": 1.4, "zone": "West Coast"},
        {"name": "Panaji", "lat": 15.4989, "lon": 73.8278, "rain_mult": 1.5, "zone": "West Coast"},
        {"name": "Thiruvananthapuram", "lat": 8.5241, "lon": 76.9366, "rain_mult": 1.3, "zone": "South West Coast"},

        # North Plains & NCR
        {"name": "Delhi", "lat": 28.6139, "lon": 77.2090, "rain_mult": 0.7, "zone": "North Plains"},
        {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462, "rain_mult": 0.8, "zone": "North Plains"},
        {"name": "Patna", "lat": 25.5941, "lon": 85.1376, "rain_mult": 0.9, "zone": "Central Plains"},

        # Northwest Arid & Semi-Arid
        {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873, "rain_mult": 0.5, "zone": "Northwest Arid"},
        {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "rain_mult": 0.6, "zone": "West Arid"},

        # Himalayas & High Altitude
        {"name": "Srinagar", "lat": 34.0837, "lon": 74.7973, "rain_mult": 0.7, "zone": "Western Himalayas"},
        {"name": "Shimla", "lat": 31.1048, "lon": 77.1734, "rain_mult": 1.1, "zone": "Western Himalayas"},
        {"name": "Leh", "lat": 34.1526, "lon": 77.5771, "rain_mult": 0.15, "zone": "Cold Desert"},

        # East Coast & Bay of Bengal (Cyclonic Zone)
        {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "rain_mult": 1.1, "zone": "East Coast"},
        {"name": "Bhubaneswar", "lat": 20.2961, "lon": 85.8245, "rain_mult": 1.3, "zone": "East Coast"},
        {"name": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185, "rain_mult": 1.0, "zone": "East Coast"},
        {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639, "rain_mult": 1.2, "zone": "Gangetic East"},

        # Northeast Hills (Extreme Precipitation)
        {"name": "Cherrapunji", "lat": 25.2986, "lon": 91.7308, "rain_mult": 2.5, "zone": "Northeast Plateau"},
        {"name": "Guwahati", "lat": 26.1445, "lon": 91.7362, "rain_mult": 1.4, "zone": "Assam Valley"},
        {"name": "Agartala", "lat": 23.8315, "lon": 91.2868, "rain_mult": 1.3, "zone": "Northeast Hills"},

        # Deccan Plateau & Central Peninsular India
        {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "rain_mult": 0.8, "zone": "South Plateau"},
        {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "rain_mult": 0.75, "zone": "Deccan Plateau"},
        {"name": "Nagpur", "lat": 21.1458, "lon": 79.0882, "rain_mult": 0.85, "zone": "Central India"},
        {"name": "Bhopal", "lat": 23.2599, "lon": 77.4126, "rain_mult": 0.85, "zone": "Central India"},
        {"name": "Raipur", "lat": 21.2514, "lon": 81.6296, "rain_mult": 0.9, "zone": "Central India"},
        {"name": "Ranchi", "lat": 23.3441, "lon": 85.3096, "rain_mult": 1.0, "zone": "Chota Nagpur"},

        # Maritime Island Region
        {"name": "Port Blair", "lat": 11.6233, "lon": 92.7265, "rain_mult": 1.6, "zone": "Bay Islands"},
    ]

    lead_times = [6, 12, 24, 48, 72, 120]

    records = []

    print(f"⏳ Generating 10-year dataset across {len(locations)} stations & {len(lead_times)} lead times...")

    for dt in dates:
        month = dt.month
        day_of_year = dt.dayofyear
        date_str = dt.strftime("%Y-%m-%d")

        # Determine Season & Weather Regime
        if month in [6, 7, 8, 9]:
            regime = "Monsoon"
            base_temp = 27.0
            base_humidity = 85.0
            base_pressure = 1002.0
            rain_prob = 0.70
            rain_scale = 16.0
        elif month in [12, 1, 2]:
            regime = "Winter"
            base_temp = 17.0
            base_humidity = 55.0
            base_pressure = 1016.0
            rain_prob = 0.12
            rain_scale = 4.0
        elif month in [3, 4, 5]:
            regime = "Summer"
            base_temp = 34.0
            base_humidity = 45.0
            base_pressure = 1007.0
            rain_prob = 0.22
            rain_scale = 9.0
        else:  # Oct, Nov
            regime = "Transition"
            base_temp = 26.0
            base_humidity = 72.0
            base_pressure = 1012.0
            rain_prob = 0.32
            rain_scale = 11.0

        for loc in locations:
            # Adjust regime parameters by geographic location zone
            loc_temp_adj = (25.0 - loc["lat"]) * 0.25
            if loc["zone"] in ["Western Himalayas", "Cold Desert"]:
                loc_temp_adj -= 14.0  # Cold alpine temperatures

            for lt in lead_times:
                temp = base_temp + loc_temp_adj + np.random.normal(0, 1.8)
                humidity = np.clip(base_humidity + np.random.normal(0, 4.5), 15.0, 100.0)
                pressure = base_pressure + np.random.normal(0, 2.5)
                u_wind = np.random.normal(2.5, 3.5)
                v_wind = np.random.normal(1.5, 3.5)

                # Ground Truth Reference Observation (ERA5 / IMD Reanalysis proxy)
                effective_prob = np.clip(rain_prob * loc["rain_mult"], 0.02, 0.95)
                is_raining = np.random.rand() < effective_prob

                if is_raining:
                    reference_rain = float(np.random.gamma(shape=1.4, scale=rain_scale * loc["rain_mult"]))
                    # Tropical Cyclones / Monsoon Extreme Events
                    if regime in ["Monsoon", "Transition"] and np.random.rand() < 0.04:
                        reference_rain += float(np.random.gamma(shape=2.5, scale=25.0))
                else:
                    reference_rain = 0.0

                reference_rain = round(max(0.0, reference_rain), 2)

                # Simulate AI Model (GraphCast / FourCastNet proxy):
                # Extremely accurate at short lead times (6-24h), degrades at 72-120h
                ai_bias = 0.1 if regime == "Monsoon" else -0.05
                ai_error_std = 0.8 + (lt / 24.0) * 1.7
                ai_forecast = max(0.0, reference_rain + np.random.normal(ai_bias, ai_error_std))

                # Simulate NWP Model (NCMRWF IMDAA / GFS proxy):
                # Physics-based, slightly higher noise at short lead times, highly stable at long lead times (72-120h)
                nwp_bias = 0.4 if reference_rain > 25.0 else 0.0
                nwp_error_std = 2.2 + (lt / 120.0) * 1.1
                nwp_forecast = max(0.0, reference_rain + np.random.normal(nwp_bias, nwp_error_std))

                # Simulate Ensemble Model (GEFS proxy)
                ensemble_error_std = 1.8 + (lt / 48.0) * 1.05
                ensemble_mean = max(0.0, reference_rain + np.random.normal(0.0, ensemble_error_std))
                ensemble_std = max(0.4, float(np.random.gamma(shape=2.0, scale=1.2 + (lt / 24.0))))

                # Historical RMSE rolling indicators
                ai_hist_rmse = round(1.8 + (lt / 24.0) * 1.4, 2)
                nwp_hist_rmse = round(3.2 + (lt / 120.0) * 0.9, 2)
                ensemble_hist_rmse = round(2.5 + (lt / 48.0) * 1.0, 2)

                records.append({
                    "date": date_str,
                    "latitude": loc["lat"],
                    "longitude": loc["lon"],
                    "month": month,
                    "day_of_year": day_of_year,
                    "lead_time_hours": lt,
                    "temperature_2m_c": round(temp, 2),
                    "humidity_pct": round(humidity, 2),
                    "surface_pressure_hpa": round(pressure, 2),
                    "u_wind_10m_ms": round(u_wind, 2),
                    "v_wind_10m_ms": round(v_wind, 2),
                    "weather_regime": regime,
                    "nwp_rainfall_mm": round(nwp_forecast, 2),
                    "ai_rainfall_mm": round(ai_forecast, 2),
                    "ensemble_mean_rainfall_mm": round(ensemble_mean, 2),
                    "ensemble_std_rainfall_mm": round(ensemble_std, 2),
                    "nwp_historical_rmse": nwp_hist_rmse,
                    "ai_historical_rmse": ai_hist_rmse,
                    "ensemble_historical_rmse": ensemble_hist_rmse,
                    "reference_rainfall_mm": reference_rain,
                })

    df = pd.DataFrame(records)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Successfully generated dataset with {len(df):,} records saved to: {output_path}")
    return df

if __name__ == "__main__":
    out_dir = "/Users/tejastiwari/.gemini/antigravity/scratch/sih26081-hybrid-forecast/data"
    out_csv = os.path.join(out_dir, "SIH26081_ML_training_sample.csv")
    generate_sih_weather_dataset(out_csv)
