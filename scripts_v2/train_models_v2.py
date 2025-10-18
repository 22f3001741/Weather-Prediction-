"""
==========================================================================
🌦️ train_models_v2.py  —  Version 2 Weather Model Trainer (TCN + XGBoost)
==========================================================================
✅ Includes feels_like (apparent_temperature) as input feature
✅ Fixes categorical features ('season', 'day_of_week')
✅ Saves models → models_v2/
✅ Saves scalers → scalers_v2/
✅ Safe: overwrites old ones cleanly for updated schema
==========================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ============================================================
# Directories
# ============================================================
MODEL_DIR = "models_v2"
SCALER_DIR = "scalers_v2"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# ============================================================
# Config
# ============================================================
SEQUENCE_LENGTH = 24
EPOCHS = 60
BATCH_SIZE = 32
CITIES = ["Bengaluru", "Chennai", "Delhi", "Kolkata", "Mumbai"]

# ============================================================
# Load dataset
# ============================================================
print("📊 Loading data...")
df = pd.read_csv("india_weather_processed.csv")
print(f"✅ Data loaded: {df.shape} | Columns: {list(df.columns)}")

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Detect datetime
if "time" in df.columns:
    df["datetime"] = pd.to_datetime(df["time"])
elif "datetime" not in df.columns:
    raise KeyError("No datetime column found!")

# Standardize column names
rename_map = {
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
}
df.rename(columns=rename_map, inplace=True)
print(f"📋 Columns standardized to: {list(df.columns)}")

# ============================================================
# Helper: Prepare TCN sequences
# ============================================================
def prepare_tcn_sequences(df_city, seq_len=24):
    features = ["temp", "feels_like", "humidity", "wind_speed", "pressure", "clouds", "dew"]
    data = df_city[features].values
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])  # predict next temp
    return np.array(X), np.array(y)

# ============================================================
# Training loop
# ============================================================
for city in CITIES:
    print("\n======================================================================")
    print(f"🌆 TRAINING CITY: {city}")
    print("======================================================================")

    df_city = df[df["city"].str.lower() == city.lower()].copy().sort_values("datetime")

    required_features = ["temp", "feels_like", "humidity", "wind_speed", "pressure", "clouds", "dew"]
    for col in required_features:
        if col not in df_city.columns:
            raise KeyError(f"❌ Missing column '{col}' for {city}")

    # Scale features
    scaler = StandardScaler()
    df_city[required_features] = scaler.fit_transform(df_city[required_features])
    joblib.dump(scaler, os.path.join(SCALER_DIR, f"{city}_scaler.pkl"))
    print("💾 Saved TCN scaler.")

    # Prepare TCN data
    X_tcn, y_tcn = prepare_tcn_sequences(df_city, seq_len=SEQUENCE_LENGTH)
    X_train, X_val, y_train, y_val = train_test_split(X_tcn, y_tcn, test_size=0.12, shuffle=False)
    print(f"📊 TCN data: {X_train.shape}, {X_val.shape}")

    # Build TCN model
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, len(required_features))),
        Conv1D(64, 3, activation="relu"),
        Dropout(0.2),
        Conv1D(32, 3, activation="relu"),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    checkpoint_path = os.path.join(MODEL_DIR, f"tcn_{city}.h5")
    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5),
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss")
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=callbacks
    )
    print(f"✅ Saved TCN model: {checkpoint_path}")

    # ============================================================
    # XGBoost Model
    # ============================================================
    features_xgb = [
        "temp", "feels_like", "humidity", "wind_speed", "pressure", "clouds", "dew",
        "year", "month", "day", "hour", "day_of_week", "season"
    ]

    df_city = df_city.dropna(subset=features_xgb + ["temp"])

    # Encode categorical columns safely
    for col in ["season", "day_of_week"]:
        if df_city[col].dtype == "object":
            df_city[col] = df_city[col].astype("category").cat.codes

    X = df_city[features_xgb].values
    y = df_city["temp"].shift(-1).dropna()
    X = X[:-1]

    scaler_xgb = StandardScaler()
    X_scaled = scaler_xgb.fit_transform(X)
    joblib.dump(scaler_xgb, os.path.join(SCALER_DIR, f"{city}_xgb_scaler.pkl"))

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.12, shuffle=False)

    model_xgb = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42
    )

    print("🚀 Training XGBoost model...")
    model_xgb.fit(X_train, y_train)
    model_xgb.save_model(os.path.join(MODEL_DIR, f"xgb_{city}.json"))
    print(f"✅ Saved XGBoost model: {MODEL_DIR}/xgb_{city}.json")

print("\n🎯 All city models trained and saved successfully in models_v2/")
