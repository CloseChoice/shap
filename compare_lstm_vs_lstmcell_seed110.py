"""
Compare LSTM (with While loop) vs LSTMCell (manually unrolled) for seed 110.
This will help us identify where SHAP value calculations diverge.
"""

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Set seed 110 (the problematic one)
SEED = 110

def setup_seed():
    """Setup all random seeds."""
    tf.keras.backend.clear_session()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(SEED)

def create_data():
    """Create the exact same data as the test."""
    rng = np.random.default_rng(SEED)

    start_datetime = pd.to_datetime("2020-01-01 00:00:00")
    end_datetime = pd.to_datetime("2023-03-31 23:00:00")
    date_rng = pd.date_range(start=start_datetime, end=end_datetime, freq="h")

    num_samples = len(date_rng)
    num_features = 7
    data = rng.random((num_samples, num_features))
    df = pd.DataFrame(data, index=date_rng, columns=[f"X{i}" for i in range(1, num_features + 1)])

    def windowed_dataset(series=None, in_horizon=None, out_horizon=None, delay=None, batch_size=None):
        total_horizon = in_horizon + out_horizon
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(total_horizon, shift=delay, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(total_horizon))
        dataset = dataset.map(lambda window: (window[:-out_horizon, :], window[-out_horizon:, 0]))
        dataset = dataset.batch(batch_size).prefetch(1)
        return dataset

    train_size = 0.4
    train_split = int(len(df) * train_size)
    df_train = df.iloc[:train_split]

    train_dataset = windowed_dataset(series=df_train.values, in_horizon=100, out_horizon=3, delay=1, batch_size=32)

    def tensor_to_arrays(input_obj=None):
        x = list(map(lambda x: x[0], input_obj))
        y = list(map(lambda x: x[1], input_obj))
        x_ = [xtmp.numpy() for xtmp in x]
        y_ = [ytmp.numpy() for ytmp in y]
        x = np.vstack(x_)
        y = np.vstack(y_)
        return x, y

    xarr, yarr = tensor_to_arrays(input_obj=train_dataset)
    return xarr, yarr

# Create LSTM model (with While loop)
def create_lstm_model():
    """Create model with LSTM layers (uses While loop internally)."""
    setup_seed()

    input_layer = tf.keras.layers.Input(shape=(100, 7))
    lstm_layer1 = tf.keras.layers.LSTM(5, return_sequences=True)(input_layer)
    lstm_layer2 = tf.keras.layers.LSTM(5, return_sequences=True)(lstm_layer1)
    lstm_layer3 = tf.keras.layers.LSTM(5)(lstm_layer2)
    output_layer = tf.keras.layers.Dense(3)(lstm_layer3)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse", metrics=["mae"])
    return model

# Create LSTMCell model (manually unrolled, no While loop)
class UnrolledLSTMCellModel(tf.keras.Model):
    """Manually unrolled LSTM using LSTMCell (no While loop)."""

    def __init__(self):
        super().__init__()
        self.lstm1_cell = tf.keras.layers.LSTMCell(5)
        self.lstm2_cell = tf.keras.layers.LSTMCell(5)
        self.lstm3_cell = tf.keras.layers.LSTMCell(5)
        self.dense = tf.keras.layers.Dense(3)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = 100

        # Initialize states for LSTM1
        h1 = tf.zeros((batch_size, 5), dtype=tf.float32)
        c1 = tf.zeros((batch_size, 5), dtype=tf.float32)
        state1 = [h1, c1]

        # Unroll LSTM1 (return sequences)
        lstm1_outputs = []
        for t in range(seq_len):
            output1, state1 = self.lstm1_cell(inputs[:, t, :], state1)
            lstm1_outputs.append(output1)

        lstm1_sequence = tf.stack(lstm1_outputs, axis=1)

        # Initialize states for LSTM2
        h2 = tf.zeros((batch_size, 5), dtype=tf.float32)
        c2 = tf.zeros((batch_size, 5), dtype=tf.float32)
        state2 = [h2, c2]

        # Unroll LSTM2 (return sequences)
        lstm2_outputs = []
        for t in range(seq_len):
            output2, state2 = self.lstm2_cell(lstm1_sequence[:, t, :], state2)
            lstm2_outputs.append(output2)

        lstm2_sequence = tf.stack(lstm2_outputs, axis=1)

        # Initialize states for LSTM3
        h3 = tf.zeros((batch_size, 5), dtype=tf.float32)
        c3 = tf.zeros((batch_size, 5), dtype=tf.float32)
        state3 = [h3, c3]

        # Unroll LSTM3 (final output only)
        for t in range(seq_len):
            output3, state3 = self.lstm3_cell(lstm2_sequence[:, t, :], state3)

        return self.dense(output3)

def create_lstmcell_model(lstm_model):
    """Create LSTMCell model as functional model and copy weights from LSTM model."""
    lstmcell_submodel = UnrolledLSTMCellModel()

    # Build the submodel
    dummy_input = tf.zeros((1, 100, 7))
    _ = lstmcell_submodel(dummy_input)

    # Copy weights from LSTM model to LSTMCell model
    lstmcell_submodel.lstm1_cell.set_weights(lstm_model.layers[1].cell.get_weights())
    lstmcell_submodel.lstm2_cell.set_weights(lstm_model.layers[2].cell.get_weights())
    lstmcell_submodel.lstm3_cell.set_weights(lstm_model.layers[3].cell.get_weights())
    lstmcell_submodel.dense.set_weights(lstm_model.layers[4].get_weights())

    # Wrap in a functional model so it has .inputs attribute
    input_layer = tf.keras.layers.Input(shape=(100, 7))
    output_layer = lstmcell_submodel(input_layer)
    lstmcell_model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    lstmcell_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse", metrics=["mae"])
    return lstmcell_model

print("=" * 80)
print("COMPARING LSTM vs LSTMCell for Seed 110")
print("=" * 80)

# Create data
xarr, yarr = create_data()
sample_51 = xarr[51:52, :, :]
background = xarr[:100, :, :]

print(f"\nData shapes:")
print(f"  Sample 51: {sample_51.shape}")
print(f"  Background: {background.shape}")

# Create LSTM model
print("\n" + "=" * 80)
print("1. LSTM Model (with While loop)")
print("=" * 80)

lstm_model = create_lstm_model()
lstm_pred = lstm_model(sample_51).numpy()
lstm_baseline = lstm_model(background).numpy().mean(0)

print(f"\nPredictions:")
print(f"  Sample 51: {lstm_pred[0]}")
print(f"  Baseline: {lstm_baseline}")
print(f"  Difference: {lstm_pred[0] - lstm_baseline}")

# Create LSTMCell model
print("\n" + "=" * 80)
print("2. LSTMCell Model (manually unrolled, no While loop)")
print("=" * 80)

lstmcell_model = create_lstmcell_model(lstm_model)
lstmcell_pred = lstmcell_model(sample_51).numpy()
lstmcell_baseline = lstmcell_model(background).numpy().mean(0)

print(f"\nPredictions:")
print(f"  Sample 51: {lstmcell_pred[0]}")
print(f"  Baseline: {lstmcell_baseline}")
print(f"  Difference: {lstmcell_pred[0] - lstmcell_baseline}")

# Verify outputs match
print("\n" + "=" * 80)
print("3. Verifying Model Outputs Match")
print("=" * 80)

output_diff = np.abs(lstm_pred - lstmcell_pred).max()
baseline_diff = np.abs(lstm_baseline - lstmcell_baseline).max()

print(f"  Max difference in predictions: {output_diff:.10e}")
print(f"  Max difference in baselines: {baseline_diff:.10e}")

if output_diff < 1e-5 and baseline_diff < 1e-5:
    print("  ✓ Models produce identical outputs!")
else:
    print("  ✗ Warning: Models produce different outputs!")

# Compute SHAP values for both models
print("\n" + "=" * 80)
print("4. Computing SHAP Values")
print("=" * 80)

print("\n  a) LSTM model SHAP values...")
explainer_lstm = shap.DeepExplainer(lstm_model, background)
shap_lstm = explainer_lstm.shap_values(sample_51, check_additivity=False)
shap_lstm_array = shap_lstm[0]

print("\n  b) LSTMCell model SHAP values...")
explainer_lstmcell = shap.DeepExplainer(lstmcell_model, background)
shap_lstmcell = explainer_lstmcell.shap_values(sample_51, check_additivity=False)
shap_lstmcell_array = shap_lstmcell[0]

# Compare SHAP values
print("\n" + "=" * 80)
print("5. Comparing SHAP Values")
print("=" * 80)

shap_diff = np.abs(shap_lstm_array - shap_lstmcell_array)
print(f"\nSHAP value differences:")
print(f"  Max: {shap_diff.max():.10e}")
print(f"  Mean: {shap_diff.mean():.10e}")
print(f"  Std: {shap_diff.std():.10e}")

# Check additivity for both models
print("\n" + "=" * 80)
print("6. Checking SHAP Additivity")
print("=" * 80)

print("\n  LSTM model (with While loop):")
for dim in range(3):
    shap_sum = shap_lstm_array[:, :, dim].sum()
    expected_diff = lstm_pred[0, dim] - lstm_baseline[dim]
    divergence = shap_sum - expected_diff

    print(f"\n    Dimension {dim}:")
    print(f"      SHAP sum: {shap_sum:.8f}")
    print(f"      Expected: {expected_diff:.8f}")
    print(f"      Divergence: {divergence:.8f} ({abs(divergence/expected_diff)*100:.2f}%)")

print("\n  LSTMCell model (manually unrolled):")
for dim in range(3):
    shap_sum = shap_lstmcell_array[:, :, dim].sum()
    expected_diff = lstmcell_pred[0, dim] - lstmcell_baseline[dim]
    divergence = shap_sum - expected_diff

    print(f"\n    Dimension {dim}:")
    print(f"      SHAP sum: {shap_sum:.8f}")
    print(f"      Expected: {expected_diff:.8f}")
    print(f"      Divergence: {divergence:.8f} ({abs(divergence/expected_diff)*100 if expected_diff != 0 else float('inf'):.2f}%)")

# Summary
print("\n" + "=" * 80)
print("7. Summary")
print("=" * 80)

print("\nKey Findings:")
print(f"  1. Model outputs match: {'✓ Yes' if output_diff < 1e-5 else '✗ No'}")
print(f"  2. SHAP values differ by: {shap_diff.max():.6e} (max)")
if shap_diff.max() < 1e-5:
    print("     → SHAP values are essentially identical!")
else:
    print("     → SHAP values differ significantly!")

# If SHAP values differ, show where
if shap_diff.max() >= 1e-5:
    print("\n" + "=" * 80)
    print("8. Analyzing Differences")
    print("=" * 80)

    for dim in range(3):
        dim_diff = np.abs(shap_lstm_array[:, :, dim] - shap_lstmcell_array[:, :, dim])
        max_idx = np.unravel_index(dim_diff.argmax(), dim_diff.shape)

        print(f"\n  Dimension {dim}:")
        print(f"    Max diff location: timestep={max_idx[0]}, feature={max_idx[1]}")
        print(f"    LSTM SHAP value: {shap_lstm_array[max_idx[0], max_idx[1], dim]:.8f}")
        print(f"    LSTMCell SHAP value: {shap_lstmcell_array[max_idx[0], max_idx[1], dim]:.8f}")
        print(f"    Difference: {dim_diff[max_idx]:.8f}")

# Plot divergence over sequence
print("\n" + "=" * 80)
print("9. Plotting Divergence Over Sequence")
print("=" * 80)

# Create cumulative divergence plots
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle('SHAP Value Divergence Analysis: LSTM vs LSTMCell (Seed 110)', fontsize=16)

for dim in range(3):
    # Get SHAP values for this dimension
    lstm_shap = shap_lstm_array[:, :, dim]  # Shape: (100 timesteps, 7 features)
    lstmcell_shap = shap_lstmcell_array[:, :, dim]

    # Calculate difference
    diff = lstm_shap - lstmcell_shap

    # Plot 1: Cumulative sum over timesteps (summed across features)
    ax1 = axes[dim, 0]
    cumsum_per_timestep = diff.sum(axis=1)  # Sum across features for each timestep
    cumulative = np.cumsum(cumsum_per_timestep)
    ax1.plot(cumulative, 'b-', linewidth=2)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_title(f'Dim {dim}: Cumulative Divergence Over Time')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Cumulative SHAP Difference')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Per-timestep divergence (summed across features)
    ax2 = axes[dim, 1]
    ax2.plot(cumsum_per_timestep, 'g-', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_title(f'Dim {dim}: Per-Timestep Divergence')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('SHAP Difference (sum across features)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Heatmap of divergence across timesteps and features
    ax3 = axes[dim, 2]
    im = ax3.imshow(diff.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax3.set_title(f'Dim {dim}: Divergence Heatmap')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Feature')
    plt.colorbar(im, ax=ax3)

plt.tight_layout()
plot_filename = 'shap_divergence_analysis_seed110.png'
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved divergence plot to: {plot_filename}")

# Analyze pattern
print("\n" + "=" * 80)
print("10. Pattern Analysis")
print("=" * 80)

for dim in range(3):
    diff = shap_lstm_array[:, :, dim] - shap_lstmcell_array[:, :, dim]
    cumsum_per_timestep = diff.sum(axis=1)
    cumulative = np.cumsum(cumsum_per_timestep)

    # Check if it's accumulating or sporadic
    abs_cumsum = np.abs(cumsum_per_timestep)
    avg_per_step = abs_cumsum.mean()
    max_per_step = abs_cumsum.max()
    final_cumulative = abs(cumulative[-1])

    print(f"\n  Dimension {dim}:")
    print(f"    Average absolute divergence per timestep: {avg_per_step:.8f}")
    print(f"    Max absolute divergence per timestep: {max_per_step:.8f}")
    print(f"    Final cumulative divergence: {final_cumulative:.8f}")

    # Find where the biggest jumps occur
    big_jumps = np.where(abs_cumsum > avg_per_step * 3)[0]
    if len(big_jumps) > 0:
        print(f"    Timesteps with large divergence (>3x avg): {big_jumps[:10].tolist()}...")
    else:
        print(f"    No particularly large jumps detected")

    # Check if it's monotonic (always increasing or decreasing)
    if np.all(cumulative[1:] >= cumulative[:-1]):
        print(f"    Pattern: Monotonic increasing (accumulating positive error)")
    elif np.all(cumulative[1:] <= cumulative[:-1]):
        print(f"    Pattern: Monotonic decreasing (accumulating negative error)")
    else:
        print(f"    Pattern: Non-monotonic (oscillating/sporadic errors)")
