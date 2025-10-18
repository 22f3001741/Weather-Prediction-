"""
update_weather_data.py
Fetches recent hourly weather data for 5 Indian cities and updates india_weather_processed.csv.
Now includes:
✅ Computation of 'feels_like (°C)' using Steadman formula
✅ Auto-fix for timestamp format inconsistencies (T vs space)
✅ Automatic backup of old dataset before each update
"""

import os
import pandas as pd
import requests
import datetime as dt
import numpy as np
from shutil import copy2

# ===========================================================
# Configuration
# ===========================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(ROOT_DIR, "india_weather_processed.csv")
BACKUP_DIR = os.path.join(ROOT_DIR, "backups")

CITIES = {
    "Bengaluru": (12.97, 77.59),
    "Chennai": (13.08, 80.27),
    "Delhi": (28.61, 77.21),
    "Kolkata": (22.57, 88.36),
    "Mumbai": (19.07, 72.88)
}
# ===========================================================


def ensure_backup():
    """Create timestamped backup before modifying the dataset."""
    if not os.path.exists(DATA_FILE):
        return
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"india_weather_processed_backup_{timestamp}.csv")
    copy2(DATA_FILE, backup_path)
    print(f"🗂️  Backup created at: {backup_path}")


def compute_feels_like(temp_c, humidity, wind_speed):
    """Compute apparent (feels-like) temperature using Steadman formula."""
    temp_c = np.array(temp_c)
    humidity = np.array(humidity)
    wind_speed = np.array(wind_speed)
    e = (humidity / 100.0) * 6.105 * np.exp((17.27 * temp_c) / (237.7 + temp_c))
    feels_like = temp_c + 0.33 * e - 0.70 * wind_speed - 4.00
    return np.round(feels_like, 2)


def normalize_timestamps(df):
    """Standardize all timestamps in the dataframe."""
    if "time" not in df.columns:
        return df
    df["time"] = pd.to_datetime(df["time"], format="mixed", errors="coerce")
    df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def fetch_weather_data(city, lat, lon, start_date, end_date):
    """Fetch hourly weather data from Open-Meteo API (archive endpoint)."""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,apparent_temperature,relative_humidity_2m,"
        f"dew_point_2m,surface_pressure,cloud_cover,wind_speed_10m,wind_direction_10m,"
        f"wind_gusts_10m"
        f"&timezone=Asia%2FKolkata"
    )

    try:
        resp = requests.get(url, timeout=20)
        data = resp.json()
        if "hourly" not in data:
            print(f"⚠️ No hourly data returned for {city}.")
            return pd.DataFrame()
        hourly = data["hourly"]
        df = pd.DataFrame(hourly)
        df["city"] = city
        return df
    except Exception as e:
        print(f"❌ Failed to fetch data for {city}: {e}")
        return pd.DataFrame()


def process_df(df):
    """Clean and format columns for india_weather_processed.csv."""
    if df.empty:
        return df

    df.rename(columns={
        "temperature_2m": "temperature (°C)",
        "apparent_temperature": "feels_like (°C)",
        "relative_humidity_2m": "humidity (%)",
        "dew_point_2m": "dew_point (°C)",
        "surface_pressure": "pressure_msl (hPa)",
        "cloud_cover": "cloud_cover (%)",
        "wind_speed_10m": "wind_speed (m/s)",
        "wind_direction_10m": "wind_direction (°)",
        "wind_gusts_10m": "wind_gusts (m/s)",
    }, inplace=True)

    # Fix timestamp format
    df = normalize_timestamps(df)

    # Compute feels_like if missing or invalid
    if "feels_like (°C)" not in df.columns or df["feels_like (°C)"].isna().any():
        print("🧮 Computing feels-like temperatures from T, RH, and wind speed...")
        df["feels_like (°C)"] = compute_feels_like(
            df["temperature (°C)"],
            df["humidity (%)"],
            df["wind_speed (m/s)"]
        )

    # Add derived temporal features
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["season"] = df["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
        6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
        10: "Post-monsoon", 11: "Post-monsoon"
    })
    return df


def main():
    print("📂 Checking dataset...")

    if os.path.exists(DATA_FILE):
        df_old = pd.read_csv(DATA_FILE)
        df_old = normalize_timestamps(df_old)
        print(f"✅ Loaded existing dataset: {DATA_FILE}")
        print(f"   Records: {len(df_old)} | Cities: {df_old['city'].nunique()}")
    else:
        print("⚠️ No dataset found. Creating a new one...")
        df_old = pd.DataFrame()

    ensure_backup()

    new_data = []
    today = dt.date.today()
    end_date = today.strftime("%Y-%m-%d")

    for city, (lat, lon) in CITIES.items():
        print(f"\n🌆 Updating data for {city}...")
        if not df_old.empty and city in df_old["city"].unique():
            last_time = df_old[df_old["city"] == city]["datetime"].max()
            start_date = (last_time + dt.timedelta(hours=1)).date().strftime("%Y-%m-%d")
        else:
            start_date = (today - dt.timedelta(days=7)).strftime("%Y-%m-%d")

        if start_date >= end_date:
            print(f"✅ {city} already up to date.")
            continue

        print(f"⏳ Fetching {start_date} → {end_date} ...")
        df_new = fetch_weather_data(city, lat, lon, start_date, end_date)
        df_new = process_df(df_new)
        if not df_new.empty:
            print(f"   ✅ New records fetched: {len(df_new)}")
            new_data.append(df_new)
        else:
            print(f"   ⚠️ No new data for {city}.")

    if not new_data:
        print("\n✅ All cities are already up to date.")
        return

    df_new_all = pd.concat(new_data, ignore_index=True)
    print("\n📦 Merging new data with existing dataset...")

    if not df_old.empty:
        df_combined = pd.concat([df_old, df_new_all], ignore_index=True)
        df_combined.drop_duplicates(subset=["city", "time"], inplace=True)
    else:
        df_combined = df_new_all

    df_combined.sort_values(["city", "datetime"], inplace=True)
    df_combined.reset_index(drop=True, inplace=True)
    df_combined = normalize_timestamps(df_combined)

    df_combined.to_csv(DATA_FILE, index=False)
    print(f"✅ Dataset updated and saved: {DATA_FILE}")
    print(f"📈 Total records now: {len(df_combined)}")


if __name__ == "__main__":
    main()
