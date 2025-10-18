import requests
import pandas as pd
from datetime import datetime
import os
import time

API_KEY = "a6a4628214c32352bb2c057d00365a9e"

def get_weather(city, retries=3):
    """
    Fetch weather data for a city with retry logic
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "city": city,
                "temperature (°C)": data["main"]["temp"],
                "feels_like (°C)": data["main"]["feels_like"],
                "temp_min (°C)": data["main"]["temp_min"],
                "temp_max (°C)": data["main"]["temp_max"],
                "humidity (%)": data["main"]["humidity"],
                "pressure (hPa)": data["main"]["pressure"],
                "weather": data["weather"][0]["description"],
                "weather_main": data["weather"][0]["main"],
                "wind_speed (m/s)": data["wind"]["speed"],
                "wind_deg": data["wind"].get("deg", 0),
                "clouds (%)": data["clouds"]["all"],
                "visibility (m)": data.get("visibility", 0),
                "sunrise": datetime.fromtimestamp(data["sys"]["sunrise"]).strftime("%Y-%m-%d %H:%M:%S"),
                "sunset": datetime.fromtimestamp(data["sys"]["sunset"]).strftime("%Y-%m-%d %H:%M:%S"),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Attempt {attempt + 1} failed for {city}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"❌ Failed to fetch data for {city} after {retries} attempts")
                return None
        
        except KeyError as e:
            print(f"❌ Data parsing error for {city}: {e}")
            return None

def collect_weather_data():
    """
    Collect weather data for all cities and save to CSV
    """
    cities = ["Chennai", "Delhi", "Mumbai", "Kolkata", "Bengaluru", 
              "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]
    
    print(f"\n{'='*60}")
    print(f"🌤️  Collecting Weather Data - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    weather_data = []
    successful = 0
    failed = 0
    
    for city in cities:
        print(f"📍 Fetching data for {city}...", end=" ")
        data = get_weather(city)
        if data:
            weather_data.append(data)
            print("✅ Success")
            successful += 1
        else:
            print("❌ Failed")
            failed += 1
    
    if not weather_data:
        print("❌ No data collected. Exiting...")
        return False
    
    df_new = pd.DataFrame(weather_data)
    
    # Append to existing CSV or create new
    csv_file = "india_weather_realtime.csv"
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        print(f"\n📊 Total records: {len(df_existing)} → {len(df_combined)}")
    else:
        df_combined = df_new
        print(f"\n📊 Created new dataset with {len(df_combined)} records")
    
    df_combined.to_csv(csv_file, index=False)
    
    print(f"\n✅ Collection Summary:")
    print(f"   Successful: {successful}/{len(cities)}")
    print(f"   Failed: {failed}/{len(cities)}")
    print(f"{'='*60}\n")
    
    return True

if __name__ == "__main__":
    collect_weather_data()