"""Debug script to understand LSTM vs LSTMCell behavior with SHAP."""

import numpy as np
import tensorflow as tf
import shap

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create synthetic data
sequence_length = 20
num_features = 10
batch_size = 50

X_train = np.random.randn(batch_size, sequence_length, num_features).astype('float32')
X_test = np.random.randn(10, sequence_length, num_features).astype('float32')

print("=" * 80)
print("Testing different LSTM configurations with SHAP DeepExplainer")
print("=" * 80)

# ============================================================================
# Test 1: Simple single LSTM layer (baseline)
# ============================================================================
print("\n1. Testing single LSTM layer...")
try:
    model_single = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=(sequence_length, num_features)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model_single.compile(loss='binary_crossentropy', optimizer='adam')

    background = X_train[:3]
    testx = X_test[0:1]

    e = shap.DeepExplainer(model_single, background)
    shap_values = e.shap_values(testx)

    predicted = model_single(testx).numpy()
    expected_baseline = model_single(background).numpy().mean(0)
    sums = np.array([shap_values[i].sum() for i in range(len(shap_values))])
    diff = predicted[0, :] - expected_baseline
    divergence = np.abs(sums - diff)

    print(f"   ✓ Single LSTM works!")
    print(f"   Predicted: {predicted[0, :]}")
    print(f"   Baseline: {expected_baseline}")
    print(f"   Difference: {diff}")
    print(f"   SHAP sum: {sums}")
    print(f"   Divergence: {divergence}")

except Exception as ex:
    print(f"   ✗ Single LSTM failed: {ex}")

# ============================================================================
# Test 2: Stacked LSTM layers (the problematic case)
# ============================================================================
print("\n2. Testing stacked LSTM layers...")
try:
    model_stacked = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(16, return_sequences=True, input_shape=(sequence_length, num_features)),
        tf.keras.layers.LSTM(8),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model_stacked.compile(loss='binary_crossentropy', optimizer='adam')

    background = X_train[:3]
    testx = X_test[0:1]

    e = shap.DeepExplainer(model_stacked, background)
    shap_values = e.shap_values(testx)

    predicted = model_stacked(testx).numpy()
    expected_baseline = model_stacked(background).numpy().mean(0)
    sums = np.array([shap_values[i].sum() for i in range(len(shap_values))])
    diff = predicted[0, :] - expected_baseline
    divergence = np.abs(sums - diff)

    print(f"   ✓ Stacked LSTM works!")
    print(f"   Predicted: {predicted[0, :]}")
    print(f"   Baseline: {expected_baseline}")
    print(f"   Difference: {diff}")
    print(f"   SHAP sum: {sums}")
    print(f"   Divergence: {divergence}")

except Exception as ex:
    print(f"   ✗ Stacked LSTM failed: {type(ex).__name__}: {str(ex)[:200]}")

# ============================================================================
# Test 3: Manual LSTMCell implementation (unrolled)
# ============================================================================
print("\n3. Testing manual LSTMCell implementation (unrolled)...")
try:
    # Create a model using LSTMCell in an unrolled manner
    class UnrolledLSTMModel(tf.keras.Model):
        def __init__(self, units, sequence_length):
            super().__init__()
            self.units = units
            self.sequence_length = sequence_length
            self.lstm_cell = tf.keras.layers.LSTMCell(units)
            self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            # Initialize states
            state = [tf.zeros((batch_size, self.units)),
                    tf.zeros((batch_size, self.units))]

            # Unroll the LSTM manually
            for t in range(self.sequence_length):
                output, state = self.lstm_cell(inputs[:, t, :], state)

            return self.dense(output)

    model_unrolled = UnrolledLSTMModel(8, sequence_length)
    model_unrolled.compile(loss='binary_crossentropy', optimizer='adam')

    # Build the model
    _ = model_unrolled(X_train[:1])

    background = X_train[:3]
    testx = X_test[0:1]

    e = shap.DeepExplainer(model_unrolled, background)
    shap_values = e.shap_values(testx)

    predicted = model_unrolled(testx).numpy()
    expected_baseline = model_unrolled(background).numpy().mean(0)
    sums = np.array([shap_values[i].sum() for i in range(len(shap_values))])
    diff = predicted[0, :] - expected_baseline
    divergence = np.abs(sums - diff)

    print(f"   ✓ Unrolled LSTMCell works!")
    print(f"   Predicted: {predicted[0, :]}")
    print(f"   Baseline: {expected_baseline}")
    print(f"   Difference: {diff}")
    print(f"   SHAP sum: {sums}")
    print(f"   Divergence: {divergence}")

except Exception as ex:
    print(f"   ✗ Unrolled LSTMCell failed: {type(ex).__name__}: {str(ex)[:200]}")

# ============================================================================
# Test 4: Compare LSTM vs Unrolled LSTMCell outputs
# ============================================================================
print("\n4. Comparing LSTM layer vs Unrolled LSTMCell outputs...")
try:
    # Create both models with same weights
    lstm_layer_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=(sequence_length, num_features)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    class UnrolledComparison(tf.keras.Model):
        def __init__(self, lstm_layer):
            super().__init__()
            self.lstm_cell = tf.keras.layers.LSTMCell(8)
            self.sequence_length = sequence_length
            self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

            # Copy weights from LSTM layer to LSTMCell
            if lstm_layer is not None:
                self.lstm_cell.set_weights(lstm_layer.weights)

        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            state = [tf.zeros((batch_size, 8)), tf.zeros((batch_size, 8))]

            for t in range(self.sequence_length):
                output, state = self.lstm_cell(inputs[:, t, :], state)

            return self.dense(output)

    # Build and get predictions from LSTM layer
    _ = lstm_layer_model(X_test[:1])
    lstm_output = lstm_layer_model(X_test[:1]).numpy()

    # Build unrolled model with same LSTM weights
    unrolled_model = UnrolledComparison(lstm_layer_model.layers[0])
    _ = unrolled_model(X_test[:1])

    # Copy Dense layer weights too
    unrolled_model.dense.set_weights(lstm_layer_model.layers[1].get_weights())

    unrolled_output = unrolled_model(X_test[:1]).numpy()

    print(f"   LSTM layer output: {lstm_output[0]}")
    print(f"   Unrolled output: {unrolled_output[0]}")
    print(f"   Difference: {np.abs(lstm_output - unrolled_output).max()}")

    if np.allclose(lstm_output, unrolled_output, atol=1e-5):
        print(f"   ✓ Outputs match!")
    else:
        print(f"   ✗ Outputs differ significantly!")

except Exception as ex:
    print(f"   ✗ Comparison failed: {type(ex).__name__}: {str(ex)[:200]}")

print("\n" + "=" * 80)
print("Testing complete!")
print("=" * 80)
