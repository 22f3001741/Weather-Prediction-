import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_historical_weather(city_name, latitude, longitude, start_date, end_date):
    """
    Fetch historical weather data using Open-Meteo API (Free, no API key needed)
    
    Args:
        city_name: Name of the city
        latitude: City latitude
        longitude: City longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "pressure_msl",
            "surface_pressure",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m"
        ],
        "timezone": "Asia/Kolkata"
    }
    
    try:
        print(f"📍 Fetching data for {city_name}...", end=" ")
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # Parse the response
        hourly_data = data["hourly"]
        
        df = pd.DataFrame({
            "city": city_name,
            "time": pd.to_datetime(hourly_data["time"]),
            "temperature (°C)": hourly_data["temperature_2m"],
            "apparent_temperature (°C)": hourly_data["apparent_temperature"],
            "humidity (%)": hourly_data["relative_humidity_2m"],
            "dew_point (°C)": hourly_data["dew_point_2m"],
            "pressure_msl (hPa)": hourly_data["pressure_msl"],
            "surface_pressure (hPa)": hourly_data["surface_pressure"],
            "cloud_cover (%)": hourly_data["cloud_cover"],
            "wind_speed (m/s)": hourly_data["wind_speed_10m"],
            "wind_direction (°)": hourly_data["wind_direction_10m"],
            "wind_gusts (m/s)": hourly_data["wind_gusts_10m"]
        })
        
        print(f"✅ Success ({len(df):,} records)")
        return df
    
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

def get_historical_data_for_cities():
    """
    Fetch 5 YEARS of historical weather data for major Indian cities
    """
    
    # Major Indian cities with coordinates
    cities = {
        "Chennai": (13.0827, 80.2707),
        "Delhi": (28.6139, 77.2090),
        "Mumbai": (19.0760, 72.8777),
        "Kolkata": (22.5726, 88.3639),
        "Bengaluru": (12.9716, 77.5946),
        "Hyderabad": (17.3850, 78.4867),
        "Pune": (18.5204, 73.8567),
        "Ahmedabad": (23.0225, 72.5714),
        "Jaipur": (26.9124, 75.7873),
        "Lucknow": (26.8467, 80.9462)
    }
    
    # Date range: Last 5 years of hourly data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d")
    
    print("="*70)
    print("🌍 Historical Weather Data Collection - 5 YEARS")
    print("="*70)
    print(f"📅 Date Range: {start_date} to {end_date} (5 years)")
    print(f"📊 Cities: {len(cities)}")
    print(f"⏰ Data Frequency: Hourly")
    print(f"⚠️  Note: This will fetch ~438,000 records (may take 5-10 minutes)")
    print(f"☕ Grab a coffee and relax...")
    print("="*70)
    print()
    
    all_data = []
    successful = 0
    total_records = 0
    
    start_time = time.time()
    
    for idx, (city_name, (lat, lon)) in enumerate(cities.items(), 1):
        print(f"[{idx}/{len(cities)}] ", end="")
        df = fetch_historical_weather(city_name, lat, lon, start_date, end_date)
        
        if df is not None:
            all_data.append(df)
            successful += 1
            total_records += len(df)
            time.sleep(1)  # Be nice to the API
    
    elapsed_time = time.time() - start_time
    
    if not all_data:
        print("\n❌ No data collected!")
        return None
    
    # Combine all city data
    print("\n🔄 Combining data from all cities...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    filename = "india_weather_historical.csv"
    print(f"💾 Saving to {filename}...")
    combined_df.to_csv(filename, index=False)
    
    print("\n" + "="*70)
    print("✅ DATA COLLECTION COMPLETE!")
    print("="*70)
    print(f"📊 Total Records: {len(combined_df):,}")
    print(f"📁 File: {filename}")
    print(f"💾 File Size: ~{len(combined_df) * 0.0001:.1f} MB")
    print(f"🏙️ Cities: {successful}/{len(cities)}")
    print(f"📅 Date Range: {combined_df['time'].min()} to {combined_df['time'].max()}")
    print(f"📈 Records per city: ~{len(combined_df) // successful:,}")
    print(f"⏱️  Time taken: {elapsed_time/60:.1f} minutes")
    print("="*70)
    
    # Show sample statistics
    print("\n📊 Temperature Statistics Across All Cities:")
    print(combined_df.groupby('city')['temperature (°C)'].describe().round(2))
    
    print("\n✅ SUCCESS! You now have 5 YEARS of weather data ready for training!")
    print("🎯 Next step: Run exploratory data analysis and build LSTM model\n")
    
    return combined_df

if __name__ == "__main__":
    print("\n🚀 Starting 5-year historical weather data collection...\n")
    df = get_historical_data_for_cities()
    
    if df is not None:
        print("="*70)
        print("📋 WHAT'S NEXT?")
        print("="*70)
        print("1. ✅ You have: india_weather_historical.csv")
        print("2. 📊 Next: Run data analysis and visualization")
        print("3. 🧠 Then: Build and train LSTM model")
        print("4. 🤖 Finally: Add LLM for weather interpretation")
        print("="*70)