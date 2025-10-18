import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🧠 LSTM Weather Forecasting - Train All Cities")
print("="*70)

# ============= 1. LOAD DATA =============
print("\n📁 Loading processed data...")
df = pd.read_csv("india_weather_processed.csv")
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values(['city', 'time'])

cities = df['city'].unique().tolist()
print(f"🏙️ Found {len(cities)} cities: {', '.join(cities)}")

# Common configuration
SEQUENCE_LENGTH = 24
EPOCHS = 50
BATCH_SIZE = 64

# Loop over each city
for CITY in cities:
    print("\n" + "="*70)
    print(f"🏗️ Training LSTM model for: {CITY}")
    print("="*70)

    city_df = df[df['city'] == CITY].copy()
    if len(city_df) < 1000:
        print(f"⚠️ Not enough data for {CITY} (records: {len(city_df)}). Skipping.")
        continue

    # ============= 2. FEATURE SELECTION =============
    feature_cols = [
        'temperature (°C)',
        'humidity (%)',
        'pressure_msl (hPa)',
        'wind_speed (m/s)',
        'cloud_cover (%)',
        'hour',
        'day_of_week',
        'month'
    ]

    data = city_df[feature_cols].values

    # ============= 3. NORMALIZATION =============
    print("🔄 Normalizing data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Save scaler
    with open(f'{CITY}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved: {CITY}_scaler.pkl")

    # ============= 4. CREATE SEQUENCES =============
    print("📊 Creating sequences...")
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length, 0])  # predict temperature
        return np.array(X), np.array(y)

    X, y = create_sequences(data_scaled, SEQUENCE_LENGTH)
    print(f"   Sequences: {len(X):,} | X shape: {X.shape} | y shape: {y.shape}")

    # ============= 5. SPLIT DATA =============
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"   Training samples: {len(X_train):,}")
    print(f"   Testing samples: {len(X_test):,}")

    # ============= 6. BUILD MODEL =============
    print("\n🏗️ Building LSTM model...")

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("✅ Model compiled")

    # ============= 7. TRAIN MODEL =============
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(f'{CITY}_lstm_model_best.h5', monitor='val_loss', save_best_only=True, verbose=1)

    print("\n🎓 Training model (may take several minutes)...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    print(f"\n✅ Training complete for {CITY}!")

    # ============= 8. SAVE MODEL =============
    model.save(f'{CITY}_lstm_model_final.h5')
    print(f"💾 Model saved: {CITY}_lstm_model_final.h5")

    # ============= 9. EVALUATION =============
    print("\n📊 Evaluating model...")
    y_pred_scaled = model.predict(X_test, verbose=0)

    dummy = np.zeros((len(y_pred_scaled), data_scaled.shape[1]))
    dummy[:, 0] = y_pred_scaled.flatten()
    y_pred = scaler.inverse_transform(dummy)[:, 0]

    dummy[:, 0] = y_test
    y_true = scaler.inverse_transform(dummy)[:, 0]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n" + "="*70)
    print(f"📈 PERFORMANCE FOR {CITY}")
    print("="*70)
    print(f"MAE:  {mae:.2f}°C")
    print(f"RMSE: {rmse:.2f}°C")
    print(f"R²:   {r2:.4f}")
    print("="*70)

    # ============= 10. VISUALIZATIONS =============
    print("📊 Generating plots...")

    # Training history
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{CITY} - Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{CITY} - MAE During Training')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{CITY}_training_history.png', dpi=300)
    plt.close()

    # Predictions vs actual
    plt.figure(figsize=(14, 5))
    plt.plot(y_true[:1000], label='Actual')
    plt.plot(y_pred[:1000], label='Predicted', linestyle='--')
    plt.title(f'{CITY} - Temperature Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{CITY}_predictions.png', dpi=300)
    plt.close()

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Temp (°C)')
    plt.ylabel('Predicted Temp (°C)')
    plt.title(f'{CITY} - Actual vs Predicted')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{CITY}_scatter.png', dpi=300)
    plt.close()

    print(f"✅ Plots saved for {CITY}:")
    print(f"   - {CITY}_training_history.png")
    print(f"   - {CITY}_predictions.png")
    print(f"   - {CITY}_scatter.png")

print("\n" + "="*70)
print("🎯 TRAINING COMPLETE FOR ALL AVAILABLE CITIES")
print("="*70)
