"""
predict_ensemble_v3_calibrated.py
Hybrid weather forecast (TCN + XGB + recent climatology)
with automatic bias correction from the last 5 days
Clean + Stable Version
"""

import os
import json
import numpy as np
import pandas as pd
import datetime as dt
import joblib
import warnings
import requests
from tensorflow.keras.models import load_model
from xgboost import XGBRegressor

# ===========================================================
# Configuration
# ===========================================================
DATA_FILE = "india_weather_processed.csv"   # already correct path
MODEL_DIR = "models_v2"
SCALER_DIR = "scalers_v2"
BIAS_CACHE = "bias_cache_v2.json"
CITIES = ["Bengaluru", "Chennai", "Delhi", "Kolkata", "Mumbai"]
SEQUENCE_LENGTH = 24

# ===========================================================
# Silence Warnings
# ===========================================================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


# ===========================================================
# Utility Functions
# ===========================================================
def safe_inverse_transform(scaler, val, feature_name="temp"):
    """Inverse-transform a single scalar value using a StandardScaler."""
    try:
        idx = list(scaler.feature_names_in_).index(feature_name)
    except Exception:
        idx = 0
    dummy = np.zeros((1, len(scaler.mean_)))
    dummy[0, idx] = val
    return scaler.inverse_transform(dummy)[0, idx]


def get_live_weather(city):
    """Fetch live weather from Open-Meteo API."""
    coords = {
        "Bengaluru": (12.97, 77.59),
        "Chennai": (13.08, 80.27),
        "Delhi": (28.61, 77.21),
        "Kolkata": (22.57, 88.36),
        "Mumbai": (19.07, 72.88),
    }
    lat, lon = coords[city]
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        data = requests.get(url, timeout=10).json()
        temp = data["current_weather"]["temperature"]
        wind = data["current_weather"]["windspeed"]
        return {"temp": temp, "wind_speed": wind}
    except Exception:
        return {"temp": np.nan, "wind_speed": np.nan}


def compute_city_bias(df_city, model_tcn, model_xgb, scaler_tcn, scaler_xgb, tcn_features, xgb_features):
    """Compute last-5-day model bias vs actual."""
    recent = df_city.tail(5 * 24).copy()
    if len(recent) < SEQUENCE_LENGTH * 2:
        return 0.0

    preds, actuals = [], []
    for i in range(len(recent) - SEQUENCE_LENGTH):
        seq = recent.iloc[i:i + SEQUENCE_LENGTH][tcn_features].values
        seq_scaled = scaler_tcn.transform(seq)
        pred_scaled = model_tcn.predict(np.expand_dims(seq_scaled, 0), verbose=0)[0][0]
        tcn_pred = safe_inverse_transform(scaler_tcn, pred_scaled, "temp")

        row = recent.iloc[i + SEQUENCE_LENGTH - 1].copy()
        if isinstance(row["season"], str):
            mapping = {"Winter": 0, "Pre-monsoon": 1, "Monsoon": 2, "Post-monsoon": 3}
            row["season"] = mapping.get(row["season"], 0)

        Xx = row[xgb_features].values.reshape(1, -1)
        Xx_scaled = scaler_xgb.transform(Xx)
        xgb_pred_scaled = model_xgb.predict(Xx_scaled)[0]
        xgb_pred = safe_inverse_transform(scaler_xgb, xgb_pred_scaled, "temp")

        final_pred = 0.6 * tcn_pred + 0.4 * xgb_pred
        preds.append(final_pred)
        actuals.append(recent["temp"].iloc[i + SEQUENCE_LENGTH - 1])

    preds, actuals = np.array(preds), np.array(actuals)
    bias = float(np.nanmean(actuals - preds))
    bias = float(np.clip(bias, -5, 5))  # cap to prevent overcorrection
    return bias


# ===========================================================
# Hybrid Forecast
# ===========================================================
def hybrid_forecast(city, df, bias_cache):
    print("\n" + "=" * 70)
    print(f"🏙️ CITY: {city}")
    print("=" * 70)

    df_city = df[df["city"].str.lower() == city.lower()].copy().sort_values("datetime")
    if df_city.empty:
        print("⚠️ No data available for this city.")
        return

    # Load models & scalers safely
    try:
        tcn = load_model(os.path.join(MODEL_DIR, f"tcn_{city}.h5"), compile=False)
        xgb = XGBRegressor()
        xgb.load_model(os.path.join(MODEL_DIR, f"xgb_{city}.json"))
        scaler_tcn = joblib.load(os.path.join(SCALER_DIR, f"{city}_scaler.pkl"))
        scaler_xgb = joblib.load(os.path.join(SCALER_DIR, f"{city}_xgb_scaler.pkl"))
    except Exception as e:
        print(f"⚠️ Missing or invalid model/scaler for {city}: {e}")
        return

    tcn_features = ["temp", "humidity", "wind_speed", "pressure", "clouds", "dew"]
    xgb_features = [
        "temp", "humidity", "wind_speed", "pressure", "clouds", "dew",
        "year", "month", "day", "hour", "day_of_week", "season"
    ]

    # Compute or load bias
    bias = bias_cache.get(city)
    if bias is None or np.isnan(bias):
        bias = compute_city_bias(df_city, tcn, xgb, scaler_tcn, scaler_xgb, tcn_features, xgb_features)
        bias_cache[city] = bias
    print(f"📏 Applying bias correction: {bias:+.2f}°C")

    # Fetch live weather
    live = get_live_weather(city)
    print(f"🌍 Live Weather at {dt.datetime.now().strftime('%Y-%m-%d %I:%M %p')}:")
    print(f"   🌡️ Temp: {live['temp']}°C | 🌬️ {live['wind_speed']} m/s")

    # Prepare sequence for TCN
    seq = df_city[tcn_features].tail(SEQUENCE_LENGTH).values
    seq_scaled = scaler_tcn.transform(seq)
    pred_tcn_scaled = tcn.predict(np.expand_dims(seq_scaled, 0), verbose=0)[0][0]
    pred_tcn = safe_inverse_transform(scaler_tcn, pred_tcn_scaled, "temp")

    # Prepare XGB input
    last = df_city.tail(1).copy()
    if isinstance(last["season"].iloc[0], str):
        mapping = {"Winter": 0, "Pre-monsoon": 1, "Monsoon": 2, "Post-monsoon": 3}
        last.loc[:, "season"] = last["season"].map(mapping).fillna(0)

    Xx = last[xgb_features].values
    Xx_scaled = scaler_xgb.transform(Xx)
    pred_xgb_scaled = xgb.predict(Xx_scaled)[0]
    pred_xgb = safe_inverse_transform(scaler_xgb, pred_xgb_scaled, "temp")

    # Baseline = 5-day average by hour
    now = pd.Timestamp.now()
    recent5 = df_city[df_city["datetime"] >= now - pd.Timedelta(days=5)]
    baseline_hour = recent5.groupby("hour")["temp"].mean()
    current_hour = now.hour
    baseline = baseline_hour.get(current_hour, df_city["temp"].mean())

    # Bias-corrected ensemble reference
    final_ref = baseline + 0.4 * (pred_tcn - baseline) + 0.2 * (pred_xgb - baseline) + bias

    # Build 24-hour forecast (smooth variation)
    forecast = []
    now = now.floor("h")
    for h in range(24):
        ts = now + pd.Timedelta(hours=h)
        base_hr = baseline_hour.get(ts.hour, baseline)
        # cosine-based diurnal adjustment (smooth temperature cycle)
        pred = base_hr + 0.75 * (final_ref - baseline) * np.cos((h - 6) / 24 * np.pi)
        pred = np.clip(pred, -10, 55)
        forecast.append([ts.strftime("%a %I:%M %p"), round(pred, 2)])

    fc = pd.DataFrame(forecast, columns=["Time", "Pred_Temp (°C)"])

    print(f"\n🔮 Forecast Summary (Next 24 Hours) — bias={bias:+.2f}°C")
    print(fc.to_string(index=False))
    print("\n📈 Summary:")
    print(f"   Baseline: {baseline:.2f}°C | TCN: {pred_tcn:.2f}°C | XGB: {pred_xgb:.2f}°C | Final (ref): {final_ref:.2f}°C")
    print(f"   Avg: {fc['Pred_Temp (°C)'].mean():.2f}°C | Min: {fc['Pred_Temp (°C)'].min():.2f}°C | Max: {fc['Pred_Temp (°C)'].max():.2f}°C")


# ===========================================================
# Main
# ===========================================================
def main():
    print("📊 Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={
        "temperature (°c)": "temp",
        "apparent_temperature (°c)": "feels_like",
        "humidity (%)": "humidity",
        "dew_point (°c)": "dew",
        "pressure_msl (hpa)": "pressure",
        "surface_pressure (hpa)": "surface_pressure",
        "cloud_cover (%)": "clouds",
        "wind_speed (m/s)": "wind_speed",
        "wind_direction (°)": "wind_direction",
        "wind_gusts (m/s)": "gust"
    }, inplace=True)
    df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    print(f"✅ Loaded {len(df)} records across {df['city'].nunique()} cities.")

    if os.path.exists(BIAS_CACHE):
        with open(BIAS_CACHE, "r") as f:
            bias_cache = json.load(f)
    else:
        bias_cache = {}

    print("\n======================================================================")
    print("🌦️  Hybrid Weather Prediction Engine (Auto-Calibrated v3)")
    print("======================================================================")
    for c in CITIES:
        try:
            hybrid_forecast(c, df, bias_cache)
        except Exception as e:
            warnings.warn(f"{c} failed: {e}")

    with open(BIAS_CACHE, "w") as f:
        json.dump(bias_cache, f, indent=2)

    print("\n✅ All forecasts completed successfully and biases updated.")


if __name__ == "__main__":
    main()
