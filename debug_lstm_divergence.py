"""
Debug script to identify where SHAP divergence occurs with stacked LSTMs.

This script compares:
1. Stacked LSTM layers (the problematic case)
2. Manual unrolled LSTMCell implementation (should work correctly)

We verify outputs match and debug SHAP value computation at each step.
"""

import numpy as np
import tensorflow as tf
import shap

print("=" * 80)
print("LSTM vs LSTMCell SHAP Divergence Debugging")
print("=" * 80)

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration
SEQUENCE_LENGTH = 20
NUM_FEATURES = 7
BATCH_SIZE = 100
NUM_BACKGROUND = 10
NUM_TEST = 5
LSTM_UNITS_1 = 16
LSTM_UNITS_2 = 8
OUTPUT_DIM = 3

print(f"\nConfiguration:")
print(f"  Sequence length: {SEQUENCE_LENGTH}")
print(f"  Num features: {NUM_FEATURES}")
print(f"  LSTM1 units: {LSTM_UNITS_1}")
print(f"  LSTM2 units: {LSTM_UNITS_2}")
print(f"  Output dim: {OUTPUT_DIM}")

# Generate data
X_data = np.random.randn(BATCH_SIZE, SEQUENCE_LENGTH, NUM_FEATURES).astype('float32')
X_background = X_data[:NUM_BACKGROUND]
X_test = X_data[NUM_BACKGROUND:NUM_BACKGROUND + NUM_TEST]

print(f"\nData shapes:")
print(f"  Background: {X_background.shape}")
print(f"  Test: {X_test.shape}")

# ============================================================================
# Model 1: Stacked LSTM (the problematic case)
# ============================================================================
print("\n" + "=" * 80)
print("Creating stacked LSTM model...")
print("=" * 80)

stacked_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(LSTM_UNITS_1, return_sequences=True,
                        input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
    tf.keras.layers.LSTM(LSTM_UNITS_2),
    tf.keras.layers.Dense(OUTPUT_DIM)
])
stacked_model.compile(loss='mse', optimizer='adam')

# Get predictions
stacked_pred = stacked_model(X_test).numpy()
stacked_baseline = stacked_model(X_background).numpy().mean(0)

print(f"Stacked LSTM predictions: {stacked_pred[0]}")
print(f"Stacked LSTM baseline: {stacked_baseline}")
print(f"Difference: {stacked_pred[0] - stacked_baseline}")

# Get SHAP values for stacked LSTM
print("\nComputing SHAP values for stacked LSTM...")
explainer_stacked = shap.DeepExplainer(stacked_model, X_background)
shap_stacked = explainer_stacked.shap_values(X_test)

# Check additivity for each output dimension
print("\nChecking SHAP additivity for stacked LSTM:")
print(f"Number of SHAP value arrays: {len(shap_stacked)}")
print(f"SHAP values shape: {[s.shape for s in shap_stacked]}")
print(f"Test samples: {NUM_TEST}")
print(f"Output dimensions: {OUTPUT_DIM}")

# SHAP values come as a list of arrays, one per test sample for multi-output models
#  Or one array per output dimension
# Let's check which it is
if len(shap_stacked) == NUM_TEST:
    print("\n→ SHAP values organized by test sample")
    # Average divergence across all samples
    all_divergences = []
    for sample_idx in range(NUM_TEST):
        sample_shap = shap_stacked[sample_idx]  # Shape: (sequence_length, n_features, n_outputs)
        print(f"\nSample {sample_idx}:")
        for dim in range(OUTPUT_DIM):
            shap_sum = sample_shap[:, :, dim].sum()
            expected_diff = stacked_pred[sample_idx, dim] - stacked_baseline[dim]
            divergence = np.abs(shap_sum - expected_diff)
            all_divergences.append(divergence)
            print(f"  Dim {dim}: SHAP sum={shap_sum:.6f}, Expected={expected_diff:.6f}, Div={divergence:.6f}")

    print(f"\nOverall max divergence: {max(all_divergences):.6f}")
    print(f"Overall mean divergence: {np.mean(all_divergences):.6f}")

elif len(shap_stacked) == OUTPUT_DIM:
    print("\n→ SHAP values organized by output dimension")
    for dim in range(OUTPUT_DIM):
        shap_sum = shap_stacked[dim].sum(axis=tuple(range(1, shap_stacked[dim].ndim)))
        expected_diff = stacked_pred[:, dim] - stacked_baseline[dim]
        divergence = np.abs(shap_sum - expected_diff)

        print(f"\nDimension {dim}:")
        print(f"  SHAP shape: {shap_stacked[dim].shape}")
        print(f"  SHAP sum: {shap_sum}")
        print(f"  Expected diff: {expected_diff}")
        print(f"  Divergence: {divergence}")
        print(f"  Max divergence: {divergence.max():.6f}")
        print(f"  Mean divergence: {divergence.mean():.6f}")

# ============================================================================
# Model 2: Manual unrolled LSTMCell (should work correctly)
# ============================================================================
print("\n" + "=" * 80)
print("Creating manual unrolled LSTMCell model...")
print("=" * 80)

class UnrolledLSTMModel(tf.keras.Model):
    """Manually unrolled LSTM using LSTMCell for debugging."""

    def __init__(self, lstm1_units, lstm2_units, output_dim, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length
        self.lstm1_cell = tf.keras.layers.LSTMCell(lstm1_units)
        self.lstm2_cell = tf.keras.layers.LSTMCell(lstm2_units)
        self.dense = tf.keras.layers.Dense(output_dim)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        # Initialize states for first LSTM
        h1 = tf.zeros((batch_size, self.lstm1_cell.units), dtype=tf.float32)
        c1 = tf.zeros((batch_size, self.lstm1_cell.units), dtype=tf.float32)
        state1 = [h1, c1]

        # Unroll first LSTM
        lstm1_outputs = []
        for t in range(self.sequence_length):
            output1, state1 = self.lstm1_cell(inputs[:, t, :], state1)
            lstm1_outputs.append(output1)

        # Stack outputs from first LSTM
        lstm1_sequence = tf.stack(lstm1_outputs, axis=1)

        # Initialize states for second LSTM
        h2 = tf.zeros((batch_size, self.lstm2_cell.units), dtype=tf.float32)
        c2 = tf.zeros((batch_size, self.lstm2_cell.units), dtype=tf.float32)
        state2 = [h2, c2]

        # Unroll second LSTM (only return final output)
        for t in range(self.sequence_length):
            output2, state2 = self.lstm2_cell(lstm1_sequence[:, t, :], state2)

        # Apply dense layer
        return self.dense(output2)

unrolled_model = UnrolledLSTMModel(LSTM_UNITS_1, LSTM_UNITS_2, OUTPUT_DIM, SEQUENCE_LENGTH)

# Build the model
_ = unrolled_model(X_test[:1])

# Copy weights from stacked model to unrolled model for fair comparison
print("\nCopying weights from stacked LSTM to unrolled model...")
unrolled_model.lstm1_cell.set_weights(stacked_model.layers[0].cell.get_weights())
unrolled_model.lstm2_cell.set_weights(stacked_model.layers[1].cell.get_weights())
unrolled_model.dense.set_weights(stacked_model.layers[2].get_weights())

# Verify outputs match
unrolled_pred = unrolled_model(X_test).numpy()
output_diff = np.abs(stacked_pred - unrolled_pred).max()
print(f"\nOutput difference (stacked vs unrolled): {output_diff:.10f}")
if output_diff < 1e-5:
    print("✓ Outputs match perfectly!")
else:
    print("✗ Warning: Outputs differ!")

# Get SHAP values for unrolled model
print("\nComputing SHAP values for unrolled LSTMCell model...")
try:
    explainer_unrolled = shap.DeepExplainer(unrolled_model, X_background)
    shap_unrolled = explainer_unrolled.shap_values(X_test)

    # Check additivity for each output dimension
    print("\nChecking SHAP additivity for unrolled LSTMCell:")
    print(f"SHAP values shape: {[s.shape for s in shap_unrolled]}")
    unrolled_baseline = unrolled_model(X_background).numpy().mean(0)

    for dim in range(OUTPUT_DIM):
        # Sum SHAP values across all feature dimensions
        if len(shap_unrolled[dim].shape) == 3:
            shap_sum = shap_unrolled[dim].sum(axis=(1, 2))
        else:
            shap_sum = shap_unrolled[dim].sum(axis=tuple(range(1, shap_unrolled[dim].ndim)))

        expected_diff = unrolled_pred[:, dim] - unrolled_baseline[dim]
        divergence = np.abs(shap_sum - expected_diff)

        print(f"\nDimension {dim}:")
        print(f"  SHAP shape: {shap_unrolled[dim].shape}")
        print(f"  SHAP sum: {shap_sum}")
        print(f"  Expected diff: {expected_diff}")
        print(f"  Divergence: {divergence}")
        print(f"  Max divergence: {divergence.max():.6f}")
        print(f"  Mean divergence: {divergence.mean():.6f}")

    # Compare SHAP values
    print("\n" + "=" * 80)
    print("Comparing SHAP values between stacked and unrolled:")
    print("=" * 80)
    for dim in range(OUTPUT_DIM):
        shap_diff = np.abs(shap_stacked[dim] - shap_unrolled[dim]).max()
        print(f"Dimension {dim} SHAP difference: {shap_diff:.10f}")

except Exception as e:
    print(f"\n✗ Error with unrolled model: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\nKey findings:")
print("1. Stacked LSTM model predictions work correctly")
print("2. SHAP values have some divergence from expected difference")
print("3. Need to investigate:")
print("   - Is the divergence due to While loop gradient handling?")
print("   - Are gradients being propagated correctly through stacked layers?")
print("   - Is _variable_inputs detecting tensors correctly in FuncGraphs?")

print("\nNext steps:")
print("1. Add more detailed logging in While loop gradient handler")
print("2. Check _variable_inputs behavior for While loop operations")
print("3. Compare gradient computation step-by-step")
