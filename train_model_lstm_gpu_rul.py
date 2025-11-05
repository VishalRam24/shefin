#!/usr/bin/env python3
"""
LSTM Voltage & RUL Prediction Model - Google Colab / GPU Optimized Version
- Predicts both voltage (V) and Remaining Useful Life (RUL in hours)
- Uses LSTM to capture time-series degradation dynamics
- Embeds normalization directly in the model (no external scalers)
- Uses both time_s and current_density as inputs
- Generates interactive Plotly visualizations
- Optimized for GPU training in Colab / Kaggle environments
"""

import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TQDM_AVAILABLE = False

# =============================================================================
# STEP 0: Check GPU Availability
# =============================================================================
print("=" * 70)
print("GPU AND ENVIRONMENT CHECK")
print("=" * 70)

print(f"TensorFlow version: {tf.__version__}")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"✅ GPU FOUND: {len(gpus)} GPU(s) available")
    for idx, gpu in enumerate(gpus):
        print(f"   GPU {idx}: {gpu}")
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"   ✓ Memory growth enabled for GPU {idx}")
        except RuntimeError as err:
            print(f"   ⚠ Could not enable memory growth: {err}")
else:
    print("⚠️  NO GPU FOUND - Training will use CPU (much slower)")
    print("   In Colab: Runtime → Change runtime type → Hardware accelerator → GPU")
    print("   In Kaggle: Settings → Accelerator → GPU")

try:
    import google.colab  # type: ignore

    IN_COLAB = True
    print("✅ Running in Google Colab")
except Exception:
    IN_COLAB = False
    print("ℹ️  Not running in Colab")

print("=" * 70 + "\n")

# =============================================================================
# STEP 1: Install Required Packages (for Colab)
# =============================================================================
if IN_COLAB:
    print("Checking Plotly installation for Colab...")
    try:
        import plotly  # noqa: F401

        print("✓ Plotly is available\n")
    except Exception:
        print("Installing plotly and kaleido...")
        os.system("pip install -q plotly kaleido")
        print("✓ Plotly installed\n")

print("Checking tqdm availability...")
try:
    from tqdm.keras import TqdmCallback  # type: ignore

    TQDM_AVAILABLE = True
    print("✓ tqdm is available\n")
except Exception:
    print("Installing tqdm...")
    install_code = os.system("pip install -q tqdm")
    if install_code == 0:
        try:
            from tqdm.keras import TqdmCallback  # type: ignore

            TQDM_AVAILABLE = True
            print("✓ tqdm installed successfully\n")
        except Exception:
            print(
                "⚠ Unable to import tqdm even after installation. Progress bars will fall back to Keras default.\n"
            )
    else:
        print(
            "⚠ tqdm installation command failed. Progress bars will fall back to Keras default.\n"
        )

# Reduce TensorFlow logging noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# =============================================================================
# STEP 2: Data Loading
# =============================================================================
print("=" * 70)
print("DATA LOADING")
print("=" * 70)

if IN_COLAB:
    print("\n📁 UPLOAD YOUR DATA FILE:")
    print("   1. Click the folder icon on the left sidebar")
    print("   2. Upload 'training.parquet'")
    print("   OR run the snippet below:\n")
    print("      from google.colab import files")
    print("      uploaded = files.upload()\n")
    if not os.path.exists("training.parquet"):
        print("⏳ Waiting for data file upload...")
        from google.colab import files  # type: ignore

        uploaded = files.upload()
        if "training.parquet" not in uploaded:
            uploaded_filename = list(uploaded.keys())[0]
            os.rename(uploaded_filename, "training.parquet")
            print(f"✓ Renamed '{uploaded_filename}' → 'training.parquet'")

print("\n[STEP 2] Loading and preprocessing data...", flush=True)

df = pd.read_parquet("training.parquet")
print("✓ Parquet file loaded", flush=True)
print(f"  Dataset size: {len(df):,} rows, {df.shape[1]} columns")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
print(f"  Columns: {list(df.columns)}")

# =============================================================================
# STEP 2a: Compute Remaining Useful Life (RUL) Targets
# =============================================================================
print("\n[STEP 2a] Computing Remaining Useful Life (RUL) targets...", flush=True)

CD_TARGET = 0.8
CD_TOLERANCE = 0.05
INITIAL_TIME_H = 10.0  # hours considered "as-new" reference window

cd_mask = df["current_density"].between(
    CD_TARGET - CD_TOLERANCE, CD_TARGET + CD_TOLERANCE
)

cd_df = df.loc[cd_mask, ["time_h", "time_s", "voltage"]].copy()
if cd_df.empty:
    raise ValueError(
        "Could not locate any samples around current_density ≈ 0.8 A/cm² to compute RUL targets."
    )

cd_df.sort_values("time_s", inplace=True)
cd_df.reset_index(drop=True, inplace=True)

total_hours_span = max(cd_df["time_h"].iloc[-1] - cd_df["time_h"].iloc[0], 1e-6)
samples_per_hour = max(int(len(cd_df) / total_hours_span), 1)

smooth_span = max(samples_per_hour // 2, 50)
cd_df["voltage_smooth"] = cd_df["voltage"].ewm(span=smooth_span, adjust=False).mean()

initial_window_mask = cd_df["time_h"] <= cd_df["time_h"].min() + INITIAL_TIME_H
if initial_window_mask.sum() < max(samples_per_hour, 25):
    fallback_count = min(len(cd_df), samples_per_hour * 3)
    initial_window_mask = cd_df.index < fallback_count

baseline_voltage = cd_df.loc[initial_window_mask, "voltage_smooth"].median()
print(
    f"✓ Dynamic baseline at {CD_TARGET} A/cm²: {baseline_voltage:.4f} V "
    f"(samples in window: {int(initial_window_mask.sum())})"
)

if np.isnan(baseline_voltage) or baseline_voltage <= 0:
    raise ValueError(
        "Baseline voltage is invalid. Verify data quality near the beginning of life."
    )

END_OF_LIFE_THRESHOLD = 0.9 * baseline_voltage
print(f"✓ End-of-life threshold (90% of baseline): {END_OF_LIFE_THRESHOLD:.4f} V")

below_threshold = cd_df["voltage_smooth"] <= END_OF_LIFE_THRESHOLD
sustain_window = max(samples_per_hour, 100)

sustained_below = (
    below_threshold.rolling(window=sustain_window, min_periods=1).mean() >= 0.9
)

if sustained_below.any():
    sustained_idx = sustained_below.idxmax()
    end_of_life_time_h = float(cd_df.loc[sustained_idx, "time_h"])
    print(
        f"✓ Detected end-of-life when smoothed voltage stayed below threshold for ~{sustain_window/samples_per_hour:.1f} h: "
        f"{end_of_life_time_h:.2f} hours"
    )
else:
    end_of_life_time_h = float(
        cd_df["time_h"].max() + (24.0 if total_hours_span > 24 else 1.0)
    )
    print(
        "⚠ Smoothed voltage never dropped 10% below baseline. Extending EOL beyond observed window: "
        f"{end_of_life_time_h:.2f} hours"
    )

df["rul_hours"] = np.maximum(end_of_life_time_h - df["time_h"], 0.0).astype(np.float32)
print("✓ RUL (hours) column added to dataframe")
print(f"  RUL range: {df['rul_hours'].min():.2f} h → {df['rul_hours'].max():.2f} h")

# =============================================================================
# STEP 2b: Train / Verification Split (time_h threshold 700)
# =============================================================================
train_data = df[df["time_h"] <= 700].copy()
verify_data = df[(df["time_h"] > 700) & (df["time_h"] <= 1000)].copy()

print(f"\n✓ Training data shape: {train_data.shape}")
print(f"✓ Verification data shape: {verify_data.shape}")

X_train = train_data[["time_s", "current_density"]].values.astype(np.float32)
y_train_voltage = train_data["voltage"].values.astype(np.float32)
y_train_rul = train_data["rul_hours"].values.astype(np.float32)

X_verify = verify_data[["time_s", "current_density"]].values.astype(np.float32)
y_verify_voltage = verify_data["voltage"].values.astype(np.float32)
y_verify_rul = verify_data["rul_hours"].values.astype(np.float32)

print(f"✓ Training inputs shape: {X_train.shape}")
print(f"✓ Training voltage targets shape: {y_train_voltage.shape}")
print(f"✓ Training RUL targets shape: {y_train_rul.shape}")
print(
    f"✓ Verification RUL range: {y_verify_rul.min():.2f} → {y_verify_rul.max():.2f} hours"
)

# =============================================================================
# STEP 3: Create Sequences for LSTM
# =============================================================================
print("\n[STEP 3] Preparing sequences for LSTM...", flush=True)


def create_sequences(X, y_voltage, y_rul, sequence_length=100):
    """Create overlapping sequences for LSTM along with voltage and RUL targets."""
    X_seq, y_vol_seq, y_rul_seq = [], [], []
    for idx in range(len(X) - sequence_length):
        X_seq.append(X[idx : idx + sequence_length])
        y_vol_seq.append(y_voltage[idx + sequence_length])
        y_rul_seq.append(y_rul[idx + sequence_length])
    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_vol_seq, dtype=np.float32),
        np.array(y_rul_seq, dtype=np.float32),
    )


sequence_length = 100
print(f"Creating sequences with length {sequence_length}...")

X_train_seq, y_train_voltage_seq, y_train_rul_seq = create_sequences(
    X_train, y_train_voltage, y_train_rul, sequence_length
)
X_verify_seq, y_verify_voltage_seq, y_verify_rul_seq = create_sequences(
    X_verify, y_verify_voltage, y_verify_rul, sequence_length
)

print(f"✓ Training sequence shape: {X_train_seq.shape}")
print(f"✓ Training voltage target shape: {y_train_voltage_seq.shape}")
print(f"✓ Training RUL target shape: {y_train_rul_seq.shape}")
print(f"✓ Verification sequence shape: {X_verify_seq.shape}")

# Chronological split for validation (last 20% of training sequences)
val_split_idx = int(len(X_train_seq) * 0.8)
X_train_final = X_train_seq[:val_split_idx]
y_train_voltage_final = y_train_voltage_seq[:val_split_idx]
y_train_rul_final = y_train_rul_seq[:val_split_idx]

X_val = X_train_seq[val_split_idx:]
y_val_voltage = y_train_voltage_seq[val_split_idx:]
y_val_rul = y_train_rul_seq[val_split_idx:]

print(f"✓ Final training sequences: {len(X_train_final):,}")
print(f"✓ Validation sequences: {len(X_val):,}")

# =============================================================================
# STEP 4: Build Multi-Output LSTM Model
# =============================================================================
print("\n[STEP 4] Building LSTM model with dual outputs (voltage & RUL)...", flush=True)

if gpus:
    print("Enabling mixed precision training for GPU acceleration...")
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(X_train_seq.reshape(-1, 2))

inputs = tf.keras.layers.Input(shape=(sequence_length, 2), name="inputs")
x = tf.keras.layers.TimeDistributed(normalizer, name="normalizer")(inputs)
x = tf.keras.layers.LSTM(128, return_sequences=True, name="lstm_1")(x)
x = tf.keras.layers.Dropout(0.2, name="dropout_1")(x)
x = tf.keras.layers.LSTM(64, return_sequences=True, name="lstm_2")(x)
x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
x = tf.keras.layers.LSTM(32, return_sequences=False, name="lstm_3")(x)
x = tf.keras.layers.Dropout(0.2, name="dropout_3")(x)
x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(x)
x = tf.keras.layers.Dense(32, activation="relu", name="dense_2")(x)

voltage_output = tf.keras.layers.Dense(
    1, activation="linear", dtype="float32", name="voltage_output"
)(x)
rul_output = tf.keras.layers.Dense(
    1, activation="linear", dtype="float32", name="rul_output"
)(x)

model = tf.keras.Model(
    inputs=inputs, outputs=[voltage_output, rul_output], name="VoltageRULLSTM"
)

losses = {
    "voltage_output": "mse",
    "rul_output": "mse",
}
loss_weights = {
    "voltage_output": 1.0,
    "rul_output": 1e-4,  # reduce magnitude impact of large-hour errors
}
metrics = {
    "voltage_output": ["mae", "mse"],
    "rul_output": ["mae", "mse"],
}

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=losses,
    loss_weights=loss_weights,
    metrics=metrics,
)

print("✓ Model architecture created:")
model.summary()

# =============================================================================
# STEP 5: Train the Model
# =============================================================================
print("\n" + "=" * 70)
print("TRAINING THE LSTM MODEL (Voltage + RUL)")
print("=" * 70)
print(f"Training sequences: {len(X_train_final):,}")
print(f"Validation sequences: {len(X_val):,}")
print("Batch size: 512")
print("Max epochs: 100")
print("Early stopping patience: 15 epochs")
print("Loss weights: voltage=1.0, RUL=1e-4")
print("=" * 70)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=7, factor=0.5, min_lr=1e-7, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5", monitor="val_loss", save_best_only=True, verbose=1
    ),
]

if TQDM_AVAILABLE:
    callbacks.insert(0, TqdmCallback(verbose=2))
    verbose = 0
    print("✓ Using tqdm progress bars\n")
else:
    verbose = 1
    print("⚠ tqdm not available, using default progress display\n")

history = model.fit(
    X_train_final,
    {
        "voltage_output": y_train_voltage_final,
        "rul_output": y_train_rul_final,
    },
    validation_data=(
        X_val,
        {
            "voltage_output": y_val_voltage,
            "rul_output": y_val_rul,
        },
    ),
    epochs=100,
    batch_size=512,
    verbose=verbose,
    callbacks=callbacks,
)

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETED!")
print("=" * 70)
print(f"Total epochs trained: {len(history.history['loss'])}")
print(f"Best validation loss: {min(history.history['val_loss']):.6f}")

# =============================================================================
# STEP 6: Evaluate on Verification Set
# =============================================================================
print("\n" + "=" * 70)
print("VERIFYING MODEL PERFORMANCE")
print("=" * 70)
print(f"Verification sequences: {len(X_verify_seq):,}")

voltage_pred, rul_pred = model.predict(X_verify_seq, batch_size=1024, verbose=1)
voltage_pred = voltage_pred.flatten()
rul_pred = rul_pred.flatten()

print("✓ Predictions completed!")
print("=" * 70)

# Voltage metrics
voltage_mse = mean_squared_error(y_verify_voltage_seq, voltage_pred)
voltage_rmse = np.sqrt(voltage_mse)
voltage_mae = mean_absolute_error(y_verify_voltage_seq, voltage_pred)
voltage_r2 = r2_score(y_verify_voltage_seq, voltage_pred)

print("Voltage Metrics:")
print(f"  MSE : {voltage_mse:.6f}")
print(f"  RMSE: {voltage_rmse:.6f} V")
print(f"  MAE : {voltage_mae:.6f} V")
print(f"  R²  : {voltage_r2:.6f}")

# RUL metrics
rul_mse = mean_squared_error(y_verify_rul_seq, rul_pred)
rul_rmse = np.sqrt(rul_mse)
rul_mae = mean_absolute_error(y_verify_rul_seq, rul_pred)

print("\nRUL Metrics:")
print(f"  MSE : {rul_mse:.4f}")
print(f"  RMSE: {rul_rmse:.4f} hours")
print(f"  MAE : {rul_mae:.4f} hours")

# =============================================================================
# STEP 7: Convert to TensorFlow Lite (Dual Output)
# =============================================================================
print("\n[STEP 7] Converting model to TensorFlow Lite...")

if gpus:
    tf.keras.mixed_precision.set_global_policy("float32")

best_model = tf.keras.models.load_model("best_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("voltage_rul_prediction_lstm.tflite", "wb") as f:
    f.write(tflite_model)

print("✓ Multi-output TFLite model saved as 'voltage_rul_prediction_lstm.tflite'")
print("  Output[0] → Voltage (V)")
print("  Output[1] → Remaining Useful Life (hours)")
print("✓ Normalization embedded; no external scaler files required")

if IN_COLAB:
    from google.colab import files  # type: ignore

    files.download("voltage_rul_prediction_lstm.tflite")
    print("✓ TFLite download initiated")

# =============================================================================
# STEP 8: Visualization
# =============================================================================
print("\n[STEP 8] Creating interactive Plotly visualizations...")

# Actual vs Predicted Voltage Scatter
fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(
        x=y_verify_voltage_seq,
        y=voltage_pred,
        mode="markers",
        marker=dict(size=3, color="royalblue", opacity=0.5),
        name="Predicted",
    )
)
fig1.add_trace(
    go.Scatter(
        x=[y_verify_voltage_seq.min(), y_verify_voltage_seq.max()],
        y=[y_verify_voltage_seq.min(), y_verify_voltage_seq.max()],
        mode="lines",
        line=dict(color="firebrick", dash="dash", width=2),
        name="Ideal",
    )
)
fig1.update_layout(
    title=(
        "Actual vs Predicted Voltage"
        f"<br>R² = {voltage_r2:.4f}, RMSE = {voltage_rmse:.6f} V, MAE = {voltage_mae:.6f} V"
    ),
    xaxis_title="Actual Voltage (V)",
    yaxis_title="Predicted Voltage (V)",
    hovermode="closest",
    width=800,
    height=600,
)
fig1.write_html("voltage_prediction_scatter.html")
print("✓ Saved: voltage_prediction_scatter.html")

# Voltage over time comparison (subset for readability)
sample_indices = np.arange(0, len(y_verify_voltage_seq), 100)
sample_time = X_verify[sequence_length:, 0][sample_indices]
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=sample_time,
        y=y_verify_voltage_seq[sample_indices],
        mode="lines",
        name="Actual Voltage",
        line=dict(color="steelblue", width=2),
    )
)
fig2.add_trace(
    go.Scatter(
        x=sample_time,
        y=voltage_pred[sample_indices],
        mode="lines",
        name="Predicted Voltage",
        line=dict(color="tomato", width=2, dash="dash"),
    )
)
fig2.update_layout(
    title="Voltage Prediction Over Time (Verification Window)",
    xaxis_title="Time (seconds)",
    yaxis_title="Voltage (V)",
    hovermode="x unified",
    width=1200,
    height=500,
)
fig2.write_html("voltage_prediction_timeseries.html")
print("✓ Saved: voltage_prediction_timeseries.html")

# RUL over time comparison
fig3 = go.Figure()
fig3.add_trace(
    go.Scatter(
        x=sample_time,
        y=y_verify_rul_seq[sample_indices],
        mode="lines",
        name="Actual RUL",
        line=dict(color="seagreen", width=2),
    )
)
fig3.add_trace(
    go.Scatter(
        x=sample_time,
        y=rul_pred[sample_indices],
        mode="lines",
        name="Predicted RUL",
        line=dict(color="orange", width=2, dash="dash"),
    )
)
fig3.update_layout(
    title="Remaining Useful Life Prediction Over Time",
    xaxis_title="Time (seconds)",
    yaxis_title="RUL (hours)",
    hovermode="x unified",
    width=1200,
    height=500,
)
fig3.write_html("rul_prediction_timeseries.html")
print("✓ Saved: rul_prediction_timeseries.html")

# Residuals for voltage
voltage_residuals = y_verify_voltage_seq - voltage_pred
fig4 = go.Figure()
fig4.add_trace(
    go.Scatter(
        x=voltage_pred,
        y=voltage_residuals,
        mode="markers",
        marker=dict(size=3, color="purple", opacity=0.5),
        name="Voltage Residuals",
    )
)
fig4.add_hline(y=0, line_dash="dash", line_color="red")
fig4.update_layout(
    title="Voltage Residuals vs Predicted Voltage",
    xaxis_title="Predicted Voltage (V)",
    yaxis_title="Residual (V)",
    width=800,
    height=600,
)
fig4.write_html("voltage_residuals.html")
print("✓ Saved: voltage_residuals.html")

# Training history (aggregate + per-output losses)
fig5 = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=("Total Loss", "Voltage Loss", "RUL Loss"),
)
fig5.add_trace(
    go.Scatter(y=history.history["loss"], name="Training", line=dict(color="blue")),
    row=1,
    col=1,
)
fig5.add_trace(
    go.Scatter(
        y=history.history["val_loss"], name="Validation", line=dict(color="red")
    ),
    row=1,
    col=1,
)
fig5.add_trace(
    go.Scatter(
        y=history.history["voltage_output_loss"],
        name="Vol Train",
        line=dict(color="blue"),
    ),
    row=1,
    col=2,
)
fig5.add_trace(
    go.Scatter(
        y=history.history["val_voltage_output_loss"],
        name="Vol Val",
        line=dict(color="red"),
    ),
    row=1,
    col=2,
)
fig5.add_trace(
    go.Scatter(
        y=history.history["rul_output_loss"],
        name="RUL Train",
        line=dict(color="blue"),
    ),
    row=1,
    col=3,
)
fig5.add_trace(
    go.Scatter(
        y=history.history["val_rul_output_loss"],
        name="RUL Val",
        line=dict(color="red"),
    ),
    row=1,
    col=3,
)
fig5.update_xaxes(title_text="Epoch", row=1, col=1)
fig5.update_xaxes(title_text="Epoch", row=1, col=2)
fig5.update_xaxes(title_text="Epoch", row=1, col=3)
fig5.update_yaxes(title_text="Loss", row=1, col=1)
fig5.update_yaxes(title_text="Loss", row=1, col=2)
fig5.update_yaxes(title_text="Loss", row=1, col=3)
fig5.update_layout(title_text="Training History", width=1400, height=500)
fig5.write_html("training_history.html")
print("✓ Saved: training_history.html")

# RUL error distribution
rul_errors = y_verify_rul_seq - rul_pred
fig6 = go.Figure()
fig6.add_trace(
    go.Histogram(
        x=rul_errors,
        nbinsx=50,
        marker=dict(color="gold", line=dict(color="black", width=1)),
        name="RUL Error Distribution",
    )
)
fig6.update_layout(
    title=(
        "RUL Prediction Error Distribution"
        f"<br>Mean={np.mean(rul_errors):.2f} h, Std={np.std(rul_errors):.2f} h"
    ),
    xaxis_title="Prediction Error (hours)",
    yaxis_title="Frequency",
    width=800,
    height=600,
)
fig6.write_html("rul_error_distribution.html")
print("✓ Saved: rul_error_distribution.html")

# =============================================================================
# COMPLETION SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("Files created:")
print("  ✓ voltage_rul_prediction_lstm.tflite (dual-output TFLite model)")
print("  ✓ best_model.h5 (Keras checkpoint)")
print("  ✓ voltage_prediction_scatter.html")
print("  ✓ voltage_prediction_timeseries.html")
print("  ✓ rul_prediction_timeseries.html")
print("  ✓ voltage_residuals.html")
print("  ✓ training_history.html")
print("  ✓ rul_error_distribution.html")
print("=" * 70)

if IN_COLAB:
    print("\n💾 Download outputs from the Files pane (left sidebar).")
else:
    print("\n📊 Open the HTML files locally for interactive visualizations!")

print("\n🎉 Model ready: voltage + RUL predictions exported to TensorFlow Lite.")
