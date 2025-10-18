"""
make_predictions.py
--------------------
Generates 24-hour weather forecasts for all cities using pre-trained LSTM models.
Automatically integrates live OpenWeatherMap data and saves outputs to daily_forecast.json.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# ================= CONFIG =================
API_KEY = "a6a4628214c32352bb2c057d00365a9e"  # your key
DATA_FILE = "india_weather_processed.csv"
SEQUENCE_LENGTH = 24
OUTPUT_JSON = "daily_forecast.json"
# ==========================================

FEATURE_COLS = [
    "temperature (°C)",
    "humidity (%)",
    "pressure_msl (hPa)",
    "wind_speed (m/s)",
    "cloud_cover (%)",
    "hour",
    "day_of_week",
    "month"
]

# ---------------------------------------------------
# Safe inverse transform that handles all scaler types
# ---------------------------------------------------
def safe_inverse_transform(scaler, scaled_val, feature_name="temperature (°C)"):
    """
    Works with StandardScaler, MinMaxScaler, or wrapped objects (dict/tuple).
    Returns inverse-transformed float value for the given feature.
    """
    try:
        # unwrap if dict or tuple
        if isinstance(scaler, dict):
            if "scaler" in scaler:
                scaler = scaler["scaler"]
        elif isinstance(scaler, tuple) and len(scaler) > 0:
            scaler = scaler[0]

        # StandardScaler
        if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
            n = len(scaler.mean_)
            dummy = np.zeros((1, n))
            idx = 0
            if hasattr(scaler, "feature_names_in_"):
                try:
                    idx = list(scaler.feature_names_in_).index(feature_name)
                except Exception:
                    idx = 0
            dummy[0, idx] = scaled_val
            inv = scaler.inverse_transform(dummy)
            return float(inv[0, idx])

        # MinMaxScaler
        elif hasattr(scaler, "min_") and hasattr(scaler, "scale_"):
            n = len(scaler.min_)
            dummy = np.zeros((1, n))
            dummy[0, 0] = scaled_val
            inv = scaler.inverse_transform(dummy)
            return float(inv[0, 0])

        # RobustScaler (just in case)
        elif hasattr(scaler, "center_") and hasattr(scaler, "scale_"):
            n = len(scaler.center_)
            dummy = np.zeros((1, n))
            dummy[0, 0] = scaled_val
            inv = scaler.inverse_transform(dummy)
            return float(inv[0, 0])

        else:
            raise ValueError(f"Unsupported scaler type: {type(scaler)}")

    except Exception as e:
        raise RuntimeError(f"Scaler inverse transform failed: {e}")


# ---------------------------------------------------
# Live weather fetch from OpenWeatherMap
# ---------------------------------------------------
def get_current_weather_openweathermap(city):
    """Fetch current weather from OpenWeatherMap API."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        d = r.json()
        return {
            "temperature": d["main"]["temp"],
            "feels_like": d["main"]["feels_like"],
            "humidity": d["main"]["humidity"],
            "pressure": d["main"]["pressure"],
            "weather": d["weather"][0]["description"].title(),
            "wind_speed": d["wind"].get("speed", 0.0),
            "clouds": d.get("clouds", {}).get("all", 0),
            "time": datetime.now()
        }
    except Exception:
        return None


# ---------------------------------------------------
# Prediction for next 24 hours
# ---------------------------------------------------
def predict_next_24(city, model, scaler, city_df, current_weather=None):
    """
    Predicts next 24 hours of temperature.
    Uses last 24 timesteps + live weather anchoring.
    """
    seq = city_df[FEATURE_COLS].tail(SEQUENCE_LENGTH).values.astype(float).copy()
    now = datetime.now()
    start = now.replace(minute=0, second=0, microsecond=0)
    if now.minute != 0:
        start += timedelta(hours=1)

    if current_weather is not None:
        try:
            seq[-1, 0] = float(current_weather["temperature"])
            seq[-1, 1] = float(current_weather["humidity"])
            seq[-1, 2] = float(current_weather["pressure"])
            seq[-1, 3] = float(current_weather["wind_speed"])
            seq[-1, 4] = float(current_weather["clouds"])
            blend = 0.3
            seq[-2, 0] = (1 - blend) * seq[-2, 0] + blend * seq[-1, 0]
            seq[-2, 1] = (1 - blend) * seq[-2, 1] + blend * seq[-1, 1]
        except Exception:
            pass

    preds = []
    for h in range(24):
        scaled = scaler.transform(seq)
        X_pred = scaled.reshape(1, SEQUENCE_LENGTH, scaled.shape[1])
        pred_scaled = model.predict(X_pred, verbose=0)[0][0]
        temp = safe_inverse_transform(scaler, pred_scaled)

        pred_time = start + timedelta(hours=h)
        preds.append({"time": pred_time, "temp": float(temp)})

        # Shift window
        seq = np.roll(seq, -1, axis=0)
        seq[-1, 0] = temp
        seq[-1, 5] = pred_time.hour
        seq[-1, 6] = pred_time.weekday()
        seq[-1, 7] = pred_time.month

    return preds


# ---------------------------------------------------
# Main routine
# ---------------------------------------------------
def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found.")

    print("=" * 70)
    print("🔮 Weather Forecasting - Multi-City 24-Hour Predictions")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["city", "time"])
    cities = df["city"].unique().tolist()
    print(f"Found cities: {', '.join(cities)}")

    all_forecasts = {}

    for city in cities:
        print(f"\n🏙️  Processing {city} ...")
        model_best = f"{city}_lstm_model_best.h5"
        model_final = f"{city}_lstm_model_final.h5"
        scaler_file = f"{city}_scaler.pkl"

        model_path = model_best if os.path.exists(model_best) else model_final
        if not os.path.exists(model_path) or not os.path.exists(scaler_file):
            print(f"⚠️ Missing model/scaler for {city}, skipping.")
            continue

        try:
            model = load_model(model_path, compile=False)
            with open(scaler_file, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            print(f"❌ Error loading {city} model/scaler: {e}")
            continue

        city_df = df[df["city"] == city].copy()
        if len(city_df) < SEQUENCE_LENGTH:
            print(f"⚠️ Not enough data for {city}, skipping.")
            continue

        live = get_current_weather_openweathermap(city)
        if live:
            print(f"   🌡️ Live temp: {live['temperature']:.2f}°C, Humidity: {live['humidity']}%")
        else:
            print("   ⚠️ Live weather unavailable; using historical anchor.")

        try:
            preds = predict_next_24(city, model, scaler, city_df, current_weather=live)
        except Exception as e:
            print(f"   ❌ Prediction failed for {city}: {e}")
            continue

        out = [
            {"Hour": p["time"].strftime("%d %b %I:%M %p"), "Predicted Temp (°C)": round(p["temp"], 2)}
            for p in preds
        ]

        all_forecasts[city] = {
            "generated_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "current_temp": live["temperature"] if live else None,
            "current_weather": {
                k: (v if not isinstance(v, datetime) else v.strftime("%d %b %Y %I:%M %p"))
                for k, v in (live.items() if live else [])
            } if live else None,
            "forecast": out
        }

    output = {
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "data": all_forecasts
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print("\n✅ Saved:", OUTPUT_JSON)
    print("=" * 70)


if __name__ == "__main__":
    main()
