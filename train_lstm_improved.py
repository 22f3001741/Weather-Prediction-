import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🧠 LSTM Weather Forecasting Model - Training")
print("="*70)

# ============= 1. LOAD AND PREPARE DATA =============
print("\n📁 Loading processed data...")
df = pd.read_csv("india_weather_processed.csv")
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values(['city', 'time'])

# Select city for training
CITY = "Delhi"  # You can change this
print(f"🏙️ Training model for: {CITY}")

city_df = df[df['city'] == CITY].copy()
print(f"   Records: {len(city_df):,}")

# ============= 2. FEATURE ENGINEERING =============
print("\n⚙️ Feature engineering...")

# Select features for training
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
print(f"   Features: {len(feature_cols)}")
print(f"   Shape: {data.shape}")

# ============= 3. NORMALIZE DATA =============
print("\n🔄 Normalizing data...")
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Save scaler for later use
with open(f'{CITY}_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"   ✅ Scaler saved: {CITY}_scaler.pkl")

# ============= 4. CREATE SEQUENCES =============
print("\n📊 Creating sequences...")
SEQUENCE_LENGTH = 24  # Use last 24 hours to predict next hour

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Predict temperature (first column)
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, SEQUENCE_LENGTH)
print(f"   Sequences created: {len(X):,}")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")

# ============= 5. TRAIN/TEST SPLIT =============
print("\n✂️ Splitting data...")
# Use 80% for training, 20% for testing
split_idx = int(len(X) * 0.8)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"   Training samples: {len(X_train):,}")
print(f"   Testing samples: {len(X_test):,}")

# ============= 6. BUILD LSTM MODEL =============
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

print("\n📋 Model Summary:")
model.summary()

# ============= 7. TRAIN MODEL =============
print("\n🎓 Training model...")
print("   This may take 10-20 minutes depending on your hardware...")

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    f'{CITY}_lstm_model_best.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

print("\n✅ Training complete!")

# ============= 8. SAVE MODEL =============
model.save(f'{CITY}_lstm_model_final.h5')
print(f"💾 Model saved: {CITY}_lstm_model_final.h5")

# ============= 9. EVALUATE MODEL =============
print("\n📊 Evaluating model...")

# Predictions on test set
y_pred_scaled = model.predict(X_test, verbose=0)

# Inverse transform to get actual temperature values
# Create dummy array with same shape as original features
dummy = np.zeros((len(y_pred_scaled), data_scaled.shape[1]))
dummy[:, 0] = y_pred_scaled.flatten()
y_pred = scaler.inverse_transform(dummy)[:, 0]

dummy[:, 0] = y_test
y_true = scaler.inverse_transform(dummy)[:, 0]

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n" + "="*70)
print("📈 MODEL PERFORMANCE")
print("="*70)
print(f"Mean Absolute Error (MAE): {mae:.2f}°C")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}°C")
print(f"R² Score: {r2:.4f}")
print("="*70)

# ============= 10. VISUALIZATIONS =============
print("\n📊 Generating visualizations...")

# Training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss During Training', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
plt.title('Model MAE During Training', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{CITY}_training_history.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {CITY}_training_history.png")

# Predictions vs Actual
plt.figure(figsize=(16, 6))

# Plot first 1000 predictions for clarity
plot_range = min(1000, len(y_true))

plt.plot(y_true[:plot_range], label='Actual Temperature', linewidth=2, alpha=0.8)
plt.plot(y_pred[:plot_range], label='Predicted Temperature', linewidth=2, alpha=0.8, linestyle='--')
plt.title(f'{CITY} - Temperature Prediction (LSTM)', fontsize=14, fontweight='bold')
plt.xlabel('Time Step')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{CITY}_predictions.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {CITY}_predictions.png")

# Scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.5, s=10)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
plt.xlabel('Actual Temperature (°C)', fontsize=12)
plt.ylabel('Predicted Temperature (°C)', fontsize=12)
plt.title(f'{CITY} - Actual vs Predicted\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{CITY}_scatter.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {CITY}_scatter.png")

print("\n" + "="*70)
print("✅ LSTM MODEL TRAINING COMPLETE!")
print("="*70)
print(f"\n📁 Generated Files:")
print(f"   1. {CITY}_lstm_model_final.h5 (Final model)")
print(f"   2. {CITY}_lstm_model_best.h5 (Best model)")
print(f"   3. {CITY}_scaler.pkl (Data scaler)")
print(f"   4. {CITY}_training_history.png")
print(f"   5. {CITY}_predictions.png")
print(f"   6. {CITY}_scatter.png")
print("\n🎯 Next Step: Make predictions and add LLM interpretation!")
print("="*70)