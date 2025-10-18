# smart_predictions.py
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os
import requests
import json

# ---------- CONFIG ----------
API_KEY = "a6a4628214c32352bb2c057d00365a9e"
DATA_FILE = "india_weather_processed.csv"
HIST_FILE = "india_weather_historical.csv"
SEQUENCE_LENGTH = 24

print("=" * 70)
print("🧠 Smart Weather Forecasting - Hourly Hybrid (LSTM + Recent5d + Climatology5yr + FeelsLike)")
print("=" * 70)

# ---------- LOAD DATA ----------
print(f"\n📊 Loading weather data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)
df["time"] = pd.to_datetime(df["time"], dayfirst=True, errors="coerce", format="mixed")
df = df.sort_values(["city", "time"]).reset_index(drop=True)
cities = df["city"].unique().tolist()
print(f"✅ Loaded {len(df):,} records across {len(cities)} cities: {', '.join(cities)}")

if os.path.exists(HIST_FILE):
    hist_df = pd.read_csv(HIST_FILE)
    hist_df["time"] = pd.to_datetime(hist_df["time"], dayfirst=True, errors="coerce", format="mixed")
    print(f"📜 Loaded historical file: {HIST_FILE} ({len(hist_df):,} records)")
else:
    hist_df = df.copy()
    print("⚠️ HIST_FILE not found — using processed data as historical baseline")

# ---------- FEATURES ----------
# Match the model’s 8-feature setup (no apparent_temperature in training)
feature_cols = [
    "temperature (°C)",
    "humidity (%)",
    "pressure_msl (hPa)",
    "wind_speed (m/s)",
    "cloud_cover (%)",
    "hour",
    "day_of_week",
    "month",
]

# ---------- FEELS-LIKE CALCULATION ----------
def compute_feels_like(temp_c, humidity, wind_speed):
    """
    Estimate apparent temperature (°C) given temp, humidity, wind speed.
    Uses Heat Index for warm conditions, Wind Chill for cold.
    """
    T = temp_c
    RH = humidity
    V = wind_speed * 3.6  # convert m/s to km/h

    # Heat Index (for T >= 26°C)
    HI = (
        -8.78469475556
        + 1.61139411 * T
        + 2.33854883889 * RH
        - 0.14611605 * T * RH
        - 0.012308094 * T**2
        - 0.016424828 * RH**2
        + 0.002211732 * T**2 * RH
        + 0.00072546 * T * RH**2
        - 0.000003582 * T**2 * RH**2
    )

    # Wind Chill (for T <= 10°C)
    WC = 13.12 + 0.6215 * T - 11.37 * (V**0.16) + 0.3965 * T * (V**0.16)

    if T >= 26:
        return HI
    elif T <= 10:
        return WC
    else:
        return T

# ---------- HELPER FUNCTIONS ----------
def get_current_weather(city):
    """Fetch current weather via OpenWeatherMap API."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        d = r.json()
        return {
            "temperature": d["main"]["temp"],
            "apparent_temperature": d["main"]["feels_like"],
            "humidity": d["main"]["humidity"],
            "pressure": d["main"]["pressure"],
            "weather": d["weather"][0]["description"].title(),
            "wind_speed": d["wind"].get("speed", np.nan),
            "clouds": d["clouds"].get("all", np.nan),
            "time": datetime.now(),
        }
    except Exception as e:
        print(f"⚠️ Live fetch failed for {city}: {e}")
        return None


def get_recent_hourly_avg(city_df, target_dt, days=5):
    end = target_dt
    start = end - pd.Timedelta(days=days)
    mask = (
        (city_df["time"] >= start)
        & (city_df["time"] < end)
        & (city_df["time"].dt.hour == target_dt.hour)
    )
    vals = city_df.loc[mask, "temperature (°C)"]
    return vals.mean() if not vals.empty else None


def get_historical_hourly_avg(hist_df, city, target_dt, years=5, day_window=0):
    start_year, end_year = target_dt.year - years, target_dt.year - 1
    mask_city = hist_df["city"] == city
    mask_years = (hist_df["time"].dt.year >= start_year) & (
        hist_df["time"].dt.year <= end_year
    )
    mask_hour = hist_df["time"].dt.hour == target_dt.hour
    mask_month = hist_df["time"].dt.month == target_dt.month

    if day_window == 0:
        mask_day = hist_df["time"].dt.day == target_dt.day
    else:
        mask_day = abs(hist_df["time"].dt.day - target_dt.day) <= day_window

    subset = hist_df.loc[
        mask_city & mask_years & mask_month & mask_day & mask_hour, "temperature (°C)"
    ]
    if not subset.empty:
        return subset.mean()

    subset2 = hist_df.loc[
        mask_city & mask_years & mask_month & mask_hour, "temperature (°C)"
    ]
    if not subset2.empty:
        return subset2.mean()

    subset3 = hist_df.loc[mask_city & mask_hour, "temperature (°C)"]
    return subset3.mean() if not subset3.empty else None

# ---------- CORE PREDICTOR ----------
def predict_next_hours(city, model, scaler, city_df, current_live, hist_df, n_hours=24):
    """Predict next n hours using LSTM + climatology + live anchoring."""
    city_df = city_df.copy().reset_index(drop=True)
    seq_raw = city_df[feature_cols].tail(SEQUENCE_LENGTH).values.copy()

    # inject current weather
    seq_raw[-1, 0] = current_live["temperature"]
    seq_raw[-1, 1] = current_live["humidity"]
    seq_raw[-1, 3] = current_live["wind_speed"]
    seq_raw[-1, 4] = current_live["clouds"]
    seq_raw[-1, 5] = current_live["time"].hour
    seq_raw[-1, 6] = current_live["time"].weekday()
    seq_raw[-1, 7] = current_live["time"].month

    now = current_live["time"]
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    pred_times, lstm_preds, recents, hists, finals, feels = [], [], [], [], [], []
    prev_temp = current_live["temperature"]

    for h in range(n_hours):
        scaled_seq = scaler.transform(seq_raw)
        X = scaled_seq.reshape(1, SEQUENCE_LENGTH, len(feature_cols))
        pred_scaled = model.predict(X, verbose=0)[0][0]
        scaled_last = scaled_seq[-1].copy()
        scaled_last[0] = pred_scaled
        pred_temp = scaler.inverse_transform([scaled_last])[0][0]

        ptime = next_hour + timedelta(hours=h)
        lstm_preds.append(pred_temp)
        pred_times.append(ptime)

        recent_avg = get_recent_hourly_avg(city_df, ptime)
        hist_avg = get_historical_hourly_avg(hist_df, city, ptime)
        recents.append(recent_avg)
        hists.append(hist_avg)

        valid = {"lstm": pred_temp, "recent": recent_avg, "hist": hist_avg}
        valid = {k: v for k, v in valid.items() if v is not None}

        if len(valid) == 3:
            keys, vals = list(valid.keys()), list(valid.values())
            dist = {
                (keys[0], keys[1]): abs(vals[0] - vals[1]),
                (keys[0], keys[2]): abs(vals[0] - vals[2]),
                (keys[1], keys[2]): abs(vals[1] - vals[2]),
            }
            best_pair = min(dist, key=dist.get)
            hybrid = np.mean([valid[best_pair[0]], valid[best_pair[1]]])
        elif len(valid) == 2:
            hybrid = np.mean(list(valid.values()))
        else:
            hybrid = list(valid.values())[0] if valid else pred_temp

        final_temp = 0.6 * prev_temp + 0.4 * hybrid if h == 0 else hybrid

        if abs(final_temp - prev_temp) > 2.5:  # momentum dampening
            final_temp = 0.7 * prev_temp + 0.3 * final_temp

        finals.append(final_temp)
        prev_temp = final_temp

        # Compute predicted feels-like temperature
        humidity_est = current_live["humidity"]
        wind_est = current_live["wind_speed"]
        fl = compute_feels_like(final_temp, humidity_est, wind_est)
        feels.append(fl)

        raw_row = seq_raw[-1].copy()
        raw_row[0] = final_temp
        raw_row[5] = ptime.hour
        raw_row[6] = ptime.weekday()
        raw_row[7] = ptime.month
        seq_raw = np.roll(seq_raw, -1, axis=0)
        seq_raw[-1] = raw_row

    preds = pd.DataFrame(
        {
            "time": pred_times,
            "lstm_temperature": lstm_preds,
            "recent_5d_avg": recents,
            "hist_5yr_avg": hists,
            "final_temperature": finals,
            "predicted_feels_like": feels,
        }
    )
    preds["final_temperature"] = (
        preds["final_temperature"].rolling(window=3, min_periods=1, center=True).mean()
    )
    preds["predicted_feels_like"] = (
        preds["predicted_feels_like"].rolling(window=3, min_periods=1, center=True).mean()
    )
    return preds

# ---------- MAIN RUNNER ----------
def run_smart_predictions():
    results = {}
    for CITY in cities:
        print("\n" + "=" * 70)
        print(f"🏙️  CITY: {CITY}")
        print("=" * 70)

        model_file1 = f"{CITY}_lstm_model_best.h5"
        model_file2 = f"{CITY}_lstm_model_final.h5"
        model_path = (
            model_file1
            if os.path.exists(model_file1)
            else (model_file2 if os.path.exists(model_file2) else None)
        )
        scaler_path = f"{CITY}_scaler.pkl"
        if model_path is None or not os.path.exists(scaler_path):
            print(f"⚠️ Missing model or scaler for {CITY}. Skipping.")
            continue

        model = load_model(model_path, compile=False)
        model.compile(optimizer="adam", loss="mse")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        city_df = df[df["city"] == CITY].copy()
        if len(city_df) < SEQUENCE_LENGTH:
            print("⚠️ Not enough data for sequence. Skipping.")
            continue

        live = get_current_weather(CITY)
        if live is None:
            print("⚠️ No live data available. Skipping.")
            continue

        preds = predict_next_hours(CITY, model, scaler, city_df, live, hist_df)

        results[CITY] = {
            "generated_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "current_weather": live,
            "forecast": preds.to_dict(orient="records"),
        }

    print("\n✅ All city forecasts completed.")
    return results


if __name__ == "__main__":
    data = run_smart_predictions()
    with open("daily_forecast.json", "w") as f:
        json.dump(data, f, indent=2, default=str)
    print("💾 Saved daily_forecast.json")
