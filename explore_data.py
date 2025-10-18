import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)

print("="*70)
print("📊 Weather Data - Exploratory Data Analysis")
print("="*70)

# Load data
print("\n📁 Loading data...")
df = pd.read_csv("india_weather_historical.csv")

# Convert time to datetime
df['time'] = pd.to_datetime(df['time'])

# Basic info
print("\n" + "="*70)
print("📋 DATASET OVERVIEW")
print("="*70)
print(f"Total Records: {len(df):,}")
print(f"Cities: {df['city'].nunique()}")
print(f"Date Range: {df['time'].min()} to {df['time'].max()}")
print(f"Duration: {(df['time'].max() - df['time'].min()).days} days")
print(f"\nColumns: {list(df.columns)}")

# Check for missing values
print("\n" + "="*70)
print("🔍 MISSING VALUES CHECK")
print("="*70)
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "✅ No missing values!")

# Statistical summary
print("\n" + "="*70)
print("📈 STATISTICAL SUMMARY")
print("="*70)
print(df.describe().round(2))

# City-wise statistics
print("\n" + "="*70)
print("🏙️ CITY-WISE TEMPERATURE STATISTICS")
print("="*70)
city_stats = df.groupby('city')['temperature (°C)'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max')
]).round(2)
print(city_stats)

# Extract time features
print("\n⏰ Extracting time features...")
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['hour'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.dayofweek
df['season'] = df['month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Monsoon', 10: 'Monsoon', 11: 'Monsoon'
})

print("✅ Time features extracted!")

# Save processed data
df.to_csv("india_weather_processed.csv", index=False)
print(f"💾 Processed data saved to: india_weather_processed.csv")

# ============= VISUALIZATIONS =============

print("\n" + "="*70)
print("📊 GENERATING VISUALIZATIONS...")
print("="*70)

# 1. Temperature Distribution by City
print("1️⃣ Temperature distribution by city...")
plt.figure(figsize=(15, 6))
for i, city in enumerate(df['city'].unique(), 1):
    plt.subplot(2, 3, i)
    city_data = df[df['city'] == city]['temperature (°C)']
    plt.hist(city_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'{city}\nMean: {city_data.mean():.1f}°C', fontsize=12, fontweight='bold')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('01_temperature_distribution.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 01_temperature_distribution.png")

# 2. Temperature Trends Over Time
print("2️⃣ Temperature trends over time...")
plt.figure(figsize=(16, 8))
for city in df['city'].unique():
    city_monthly = df[df['city'] == city].groupby(df['time'].dt.to_period('M'))['temperature (°C)'].mean()
    plt.plot(city_monthly.index.to_timestamp(), city_monthly.values, label=city, linewidth=2, alpha=0.8)
plt.title('Temperature Trends (Monthly Average) - 5 Years', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('02_temperature_trends.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 02_temperature_trends.png")

# 3. Seasonal Patterns
print("3️⃣ Seasonal patterns...")
plt.figure(figsize=(14, 6))
seasonal_data = df.groupby(['city', 'season'])['temperature (°C)'].mean().reset_index()
sns.barplot(data=seasonal_data, x='season', y='temperature (°C)', hue='city', palette='Set2')
plt.title('Average Temperature by Season and City', fontsize=14, fontweight='bold')
plt.xlabel('Season', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('03_seasonal_patterns.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 03_seasonal_patterns.png")

# 4. Hourly Temperature Patterns
print("4️⃣ Hourly temperature patterns...")
plt.figure(figsize=(14, 6))
hourly_data = df.groupby(['city', 'hour'])['temperature (°C)'].mean().reset_index()
for city in df['city'].unique():
    city_hourly = hourly_data[hourly_data['city'] == city]
    plt.plot(city_hourly['hour'], city_hourly['temperature (°C)'], label=city, linewidth=2, marker='o')
plt.title('Average Temperature by Hour of Day', fontsize=14, fontweight='bold')
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.xticks(range(0, 24))
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('04_hourly_patterns.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 04_hourly_patterns.png")

# 5. Correlation Heatmap
print("5️⃣ Correlation heatmap...")
plt.figure(figsize=(12, 10))
correlation_cols = ['temperature (°C)', 'apparent_temperature (°C)', 'humidity (%)', 
                    'dew_point (°C)', 'pressure_msl (hPa)', 'wind_speed (m/s)', 'cloud_cover (%)']
corr_matrix = df[correlation_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Weather Features Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 05_correlation_heatmap.png")

# 6. Multi-feature comparison for one city
print("6️⃣ Multi-feature comparison (Delhi)...")
delhi_data = df[df['city'] == 'Delhi'][['time', 'temperature (°C)', 'humidity (%)', 'wind_speed (m/s)']].set_index('time')
delhi_monthly = delhi_data.resample('ME').mean()

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Temperature
axes[0].plot(delhi_monthly.index, delhi_monthly['temperature (°C)'], color='red', linewidth=2)
axes[0].set_title('Delhi - Temperature Over Time', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Temperature (°C)')
axes[0].grid(alpha=0.3)

# Humidity
axes[1].plot(delhi_monthly.index, delhi_monthly['humidity (%)'], color='blue', linewidth=2)
axes[1].set_title('Delhi - Humidity Over Time', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Humidity (%)')
axes[1].grid(alpha=0.3)

# Wind Speed
axes[2].plot(delhi_monthly.index, delhi_monthly['wind_speed (m/s)'], color='green', linewidth=2)
axes[2].set_title('Delhi - Wind Speed Over Time', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Wind Speed (m/s)')
axes[2].set_xlabel('Date')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('06_delhi_multifeature.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 06_delhi_multifeature.png")

print("\n" + "="*70)
print("✅ EXPLORATORY DATA ANALYSIS COMPLETE!")
print("="*70)
print("\n📊 Generated Visualizations:")
print("   1. 01_temperature_distribution.png")
print("   2. 02_temperature_trends.png")
print("   3. 03_seasonal_patterns.png")
print("   4. 04_hourly_patterns.png")
print("   5. 05_correlation_heatmap.png")
print("   6. 06_delhi_multifeature.png")
print("\n💾 Processed data: india_weather_processed.csv")
print("\n🎯 Next Step: Build LSTM model for forecasting!")
print("="*70)