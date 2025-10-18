import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

# ==============================================================
# CONFIGURATION
# ==============================================================
DATA_PATH = "india_weather_processed.csv"
OUTPUT_DIR = "models"
SCALER_DIR = "scalers"
META_DIR = "meta"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

SEQUENCE_LENGTH = 24
RANDOM_SEED = 42
CITIES_TO_TRAIN = ["Bengaluru", "Chennai", "Delhi", "Kolkata", "Mumbai"]

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ==============================================================
# DATA LOADING + COLUMN NORMALIZATION
# ==============================================================
print("📊 Loading data...")
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower() for c in df.columns]

# Auto-detect datetime column
datetime_col = None
for col in df.columns:
    if "date" in col or "time" in col:
        datetime_col = col
        break

if datetime_col is None:
    raise KeyError("❌ No datetime-like column found in dataset! Please check the CSV headers.")

df["datetime"] = pd.to_datetime(df[datetime_col])
df["month"] = df["datetime"].dt.month
df["hour"] = df["datetime"].dt.hour

# 🔹 Rename to standard names
rename_map = {
    'temperature (°c)': 'temp',
    'humidity (%)': 'humidity',
    'pressure_msl (hpa)': 'pressure',
    'wind_speed (m/s)': 'wind_speed',
    'cloud_cover (%)': 'clouds',
    'dew_point (°c)': 'dew',
    'wind_gusts (m/s)': 'gust',
    'apparent_temperature (°c)': 'feels_like',
    'surface_pressure (hpa)': 'surface_pressure'
}
df.rename(columns=rename_map, inplace=True)

print(f"✅ Data loaded: {df.shape} | Detected datetime column: '{datetime_col}'")
print("📋 Columns standardized to:", df.columns.tolist())

# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def prepare_tcn_sequences(df_city, seq_len=24):
    """Prepare sequential input for TCN model"""
    features = ['temp', 'humidity', 'pressure', 'wind_speed', 'clouds', 'dew']
    for f in features:
        if f not in df_city.columns:
            df_city[f] = 0.0  # handle missing features gracefully

    X, y = [], []
    arr = df_city[features].values
    for i in range(seq_len, len(arr)):
        X.append(arr[i-seq_len:i])
        y.append(arr[i, 0])  # predict temperature
    return np.array(X), np.array(y)


def prepare_xgb_tabular(df_city, seq_len=24):
    """Prepare lag-based tabular data for XGBoost"""
    df_city = df_city.copy()
    for lag in range(1, seq_len + 1):
        df_city[f"temp_lag{lag}"] = df_city["temp"].shift(lag)
    df_city = df_city.dropna()

    features = [col for col in df_city.columns if col.startswith("temp_lag")] + [
        "humidity", "pressure", "wind_speed", "clouds", "dew", "month", "hour"
    ]
    for f in features:
        if f not in df_city.columns:
            df_city[f] = 0.0  # fill missing safely

    X = df_city[features].values
    y = df_city["temp"].values
    return X, y


def build_tcn_model(input_shape):
    """Define a simple Temporal Convolutional Network"""
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv1D(128, 3, padding="causal", activation="relu", dilation_rate=1),
        layers.Conv1D(128, 3, padding="causal", activation="relu", dilation_rate=2),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model

# ==============================================================
# TRAINING LOOP
# ==============================================================

for city in CITIES_TO_TRAIN:
    print("\n" + "="*70)
    print(f"🌆 TRAINING CITY: {city}")
    print("="*70)

    df_city = df[df["city"].str.lower() == city.lower()].sort_values("datetime").reset_index(drop=True)

    if df_city.empty:
        print(f"⚠️ No data found for {city}! Skipping...")
        continue

    # ------------------ TCN MODEL ------------------
    X_tcn, y_tcn = prepare_tcn_sequences(df_city, seq_len=SEQUENCE_LENGTH)
    if len(X_tcn) == 0:
        print(f"⚠️ Insufficient data for TCN training in {city}. Skipping.")
        continue

    X_tr, X_te, y_tr, y_te = train_test_split(X_tcn, y_tcn, test_size=0.12, shuffle=False)

    scaler = StandardScaler()
    n_samples, seq_len, n_features = X_tr.shape
    X_tr_scaled = scaler.fit_transform(X_tr.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
    X_te_scaled = scaler.transform(X_te.reshape(-1, n_features)).reshape(X_te.shape[0], seq_len, n_features)

    joblib.dump(scaler, os.path.join(SCALER_DIR, f"{city}_scaler.pkl"))
    print("💾 Saved TCN scaler.")

    model_path = os.path.join(OUTPUT_DIR, f"tcn_{city}.h5")

    if os.path.exists(model_path):
        print(f"✅ Skipping TCN (already exists): {model_path}")
    else:
        print(f"📚 Training TCN: {X_tr.shape}, test: {X_te.shape}")
        tcn_model = build_tcn_model((SEQUENCE_LENGTH, n_features))
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
            keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss")
        ]
        history = tcn_model.fit(
            X_tr_scaled, y_tr,
            validation_data=(X_te_scaled, y_te),
            epochs=60,
            batch_size=64,
            verbose=2,
            callbacks=callbacks
        )
        print(f"✅ Saved TCN model: {model_path}")

    # ------------------ XGBOOST MODEL ------------------
    X_xgb, y_xgb = prepare_xgb_tabular(df_city, seq_len=SEQUENCE_LENGTH)
    if len(X_xgb) == 0:
        print(f"⚠️ Skipping XGB for {city} due to insufficient data.")
        continue

    xgb_scaler = StandardScaler()
    X_xgb_scaled = xgb_scaler.fit_transform(X_xgb)
    joblib.dump(xgb_scaler, os.path.join(SCALER_DIR, f"{city}_xgb_scaler.pkl"))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_xgb_scaled, y_xgb, test_size=0.12, random_state=RANDOM_SEED, shuffle=False
    )
    print("📊 XGB data prepared:", X_tr.shape, X_te.shape)

    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        tree_method="hist"
    )

    print("🚀 Training XGBoost model...")

    # ✅ Works with all XGBoost versions
    try:
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            verbose=False,
            callbacks=[xgb.callback.EarlyStopping(rounds=20, save_best=True)]
        )
    except TypeError:
        try:
            xgb_model.fit(
                X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                early_stopping_rounds=20,
                verbose=False
            )
        except TypeError:
            print("⚠️ Your XGBoost version does not support early stopping. Training without it.")
            xgb_model.fit(X_tr, y_tr)

    xgb_model.save_model(os.path.join(OUTPUT_DIR, f"xgb_{city}.json"))
    print(f"✅ Saved XGBoost model: models/xgb_{city}.json")

print("\n🎯 All city models trained and saved successfully!")
