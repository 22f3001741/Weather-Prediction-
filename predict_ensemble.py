import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import xgboost as xgb
import requests

# ===================== CONFIG =====================
DATA_PATH = "india_weather_processed.csv"
MODEL_DIR = "models"
SCALER_DIR = "scalers"
API_KEY = "a6a4628214c32352bb2c057d00365a9e"
SEQUENCE_LENGTH = 24

print("="*70)
print("🌦️  Hybrid Weather Prediction Engine (TCN + XGBoost + Climatology)")
print("="*70)

# ===================== LOAD DATA =====================
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower() for c in df.columns]

# 🔹 Rename columns for consistency
rename_map = {
    'temperature (°c)': 'temp',
    'humidity (%)': 'humidity',
    'pressure_msl (hpa)': 'pressure',
    'wind_speed (m/s)': 'wind_speed',
    'cloud_cover (%)': 'clouds',
    'dew_point (°c)': 'dew',
    'apparent_temperature (°c)': 'feels_like',
    'surface_pressure (hpa)': 'surface_pressure'
}
df.rename(columns=rename_map, inplace=True)

# 🔹 Ensure datetime column
datetime_col = None
for col in df.columns:
    if "time" in col or "date" in col:
        datetime_col = col
        break
if datetime_col is None:
    raise KeyError("❌ No time/date column found!")
df["time"] = pd.to_datetime(df[datetime_col])

df = df.sort_values(["city", "time"])
cities = df["city"].unique()
print(f"✅ Loaded {len(df):,} records across {len(cities)} cities.")
print("📋 Columns available:", list(df.columns))

# ===================== LIVE WEATHER API =====================
def get_live_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        d = r.json()
        return {
            "temp": d["main"]["temp"],
            "humidity": d["main"]["humidity"],
            "pressure": d["main"]["pressure"],
            "wind_speed": d["wind"]["speed"],
            "clouds": d["clouds"]["all"],
            "time": datetime.now()
        }
    except Exception:
        return None

# ===================== UTILITY FUNCTIONS =====================
def get_recent_5d_hourly_avg(city_df, target_time):
    start = target_time - timedelta(days=5)
    mask = (city_df["time"] >= start) & (city_df["time"] < target_time)
    if mask.sum() == 0:
        return {}
    city_df["hour"] = city_df["time"].dt.hour
    return city_df.groupby("hour")["temp"].mean().to_dict()

def get_climatology_5yr_avg(city_df, target_time):
    city_df["hour"] = city_df["time"].dt.hour
    city_df["month"] = city_df["time"].dt.month
    city_df["day"] = city_df["time"].dt.day
    mask = (city_df["month"] == target_time.month) & (city_df["day"] == target_time.day)
    if mask.sum() == 0:
        return {}
    return city_df[mask].groupby("hour")["temp"].mean().to_dict()

def create_lag_features(city_df):
    df_lag = city_df.copy()
    for lag in range(1, SEQUENCE_LENGTH + 1):
        df_lag[f"temp_lag{lag}"] = df_lag["temp"].shift(lag)
    return df_lag.dropna()

# ===================== FORECAST FUNCTION =====================
def hybrid_forecast(city, city_df):
    print("\n" + "="*70)
    print(f"🏙️  CITY: {city}")
    print("="*70)

    # Paths
    tcn_path = os.path.join(MODEL_DIR, f"tcn_{city}.h5")
    xgb_path = os.path.join(MODEL_DIR, f"xgb_{city}.json")
    scaler_path = os.path.join(SCALER_DIR, f"{city}_scaler.pkl")
    xgb_scaler_path = os.path.join(SCALER_DIR, f"{city}_xgb_scaler.pkl")

    if not (os.path.exists(tcn_path) and os.path.exists(xgb_path)):
        print("⚠️ Missing models, skipping city.")
        return

    # Load models
    tcn_model = load_model(tcn_path, compile=False)
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_path)
    scaler = joblib.load(scaler_path)
    xgb_scaler = joblib.load(xgb_scaler_path)

    # ✅ Use the exact 6 training features
    tcn_features = ["temp", "humidity", "pressure", "wind_speed", "clouds", "dew"]
    for f in tcn_features:
        if f not in city_df.columns:
            city_df[f] = 0.0

    city_df = city_df.sort_values("time")
    recent_seq = city_df.tail(SEQUENCE_LENGTH)[tcn_features].values
    start_time = city_df["time"].iloc[-1]

    hist_avg = get_climatology_5yr_avg(city_df, start_time)
    recent_avg = get_recent_5d_hourly_avg(city_df, start_time)

    # Live weather
    live = get_live_weather(city)
    if live:
        print(f"🌍 Live Weather at {live['time'].strftime('%Y-%m-%d %I:%M %p')}:")
        print(f"   🌡️ Temp: {live['temp']}°C | 💧 {live['humidity']}% | 🌬️ {live['wind_speed']} m/s | ☁️ {live['clouds']}% | ⏱ {live['pressure']} hPa")

    predictions = []
    for h in range(1, 25):
        pred_time = start_time + timedelta(hours=h)

        scaled_seq = scaler.transform(recent_seq)
        X_tcn = scaled_seq.reshape(1, SEQUENCE_LENGTH, len(tcn_features))
        tcn_pred = tcn_model.predict(X_tcn, verbose=0)[0][0]

        lag_df = create_lag_features(city_df)
        if len(lag_df) > 0:
            X_xgb = lag_df.tail(1).drop(columns=["temp"]).values
            X_xgb_scaled = xgb_scaler.transform(X_xgb)
            xgb_pred = xgb_model.predict(X_xgb_scaled)[0]
        else:
            xgb_pred = tcn_pred

        hour = pred_time.hour
        avg_5d = recent_avg.get(hour, tcn_pred)
        avg_5yr = hist_avg.get(hour, tcn_pred)

        final_pred = 0.45*tcn_pred + 0.35*xgb_pred + 0.1*avg_5d + 0.1*avg_5yr
        predictions.append((pred_time, final_pred))

        new_row = recent_seq[-1].copy()
        new_row[0] = final_pred
        recent_seq = np.vstack([recent_seq[1:], new_row])

    print("\n📊 24-HOUR FORECAST (Hybrid Ensemble)")
    print("="*70)
    for t, p in predictions:
        print(f"   {t.strftime('%Y-%m-%d %I:%M %p')} → {p:.2f}°C")

    temps = [p for _, p in predictions]
    print(f"\n📈 Summary: Avg {np.mean(temps):.2f}°C | Min {np.min(temps):.2f}°C | Max {np.max(temps):.2f}°C")

# ===================== MAIN =====================
for city in cities:
    city_df = df[df["city"].str.lower() == city.lower()].copy()
    hybrid_forecast(city, city_df)

print("\n✅ All city forecasts completed successfully!")
