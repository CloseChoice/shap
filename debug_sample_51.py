"""
Deep dive into why sample 51 with seed 110 has such high divergence.
"""

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import shap

# Set seed 110
seed = 110
tf.keras.backend.clear_session()
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
if hasattr(tf.keras.utils, "set_random_seed"):
    tf.keras.utils.set_random_seed(seed)

rng = np.random.default_rng(seed)

# Recreate the exact test scenario
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
valid_size = 0.5
train_split = int(len(df) * train_size)
valid_split = int(len(df) * (train_size + valid_size))

df_train = df.iloc[:train_split]
n_features = df.shape[1]
train_dataset = windowed_dataset(series=df_train.values, in_horizon=100, out_horizon=3, delay=1, batch_size=32)

# Build model
input_layer = tf.keras.layers.Input(shape=(100, n_features))
lstm_layer1 = tf.keras.layers.LSTM(5, return_sequences=True)(input_layer)
lstm_layer2 = tf.keras.layers.LSTM(5, return_sequences=True)(lstm_layer1)
lstm_layer3 = tf.keras.layers.LSTM(5)(lstm_layer2)
output_layer = tf.keras.layers.Dense(3)(lstm_layer3)
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse", metrics=["mae"])

def tensor_to_arrays(input_obj=None):
    x = list(map(lambda x: x[0], input_obj))
    y = list(map(lambda x: x[1], input_obj))
    x_ = [xtmp.numpy() for xtmp in x]
    y_ = [ytmp.numpy() for ytmp in y]
    x = np.vstack(x_)
    y = np.vstack(y_)
    return x, y

xarr, yarr = tensor_to_arrays(input_obj=train_dataset)

print("=" * 80)
print("Analyzing Sample 51 with Seed 110")
print("=" * 80)

# Get the problematic sample
sample_51 = xarr[51:52, :, :]  # Shape: (1, 100, 7)
background = xarr[:100, :, :]

print(f"\nSample 51 input shape: {sample_51.shape}")
print(f"Background shape: {background.shape}")

# Model predictions
pred_sample = model(sample_51).numpy()
pred_background = model(background).numpy()
expected_baseline = pred_background.mean(0)

print(f"\nModel predictions:")
print(f"  Sample 51: {pred_sample[0]}")
print(f"  Baseline (mean of 100): {expected_baseline}")
print(f"  Difference: {pred_sample[0] - expected_baseline}")

# SHAP analysis
print(f"\nComputing SHAP values...")
e = shap.DeepExplainer(model, background)
sv = e.shap_values(sample_51, check_additivity=False)

print(f"\nSHAP value structure:")
print(f"  Type: {type(sv)}")
print(f"  Length: {len(sv)}")
if len(sv) > 0:
    print(f"  First element shape: {sv[0].shape}")

# For single sample, SHAP values come as list with one element per sample
# Shape is (sequence_length, features, n_outputs)
shap_array = sv[0]  # Get first (and only) sample
print(f"\nSHAP array shape: {shap_array.shape}")

print(f"\nSHAP value statistics:")
for dim in range(3):
    # Sum across sequence and features for this output dimension
    shap_sum = shap_array[:, :, dim].sum()
    expected_diff = pred_sample[0, dim] - expected_baseline[dim]
    divergence = shap_sum - expected_diff

    print(f"\n  Dimension {dim}:")
    print(f"    SHAP sum: {shap_sum:.8f}")
    print(f"    Expected diff: {expected_diff:.8f}")
    print(f"    Divergence: {divergence:.8f}")
    print(f"    Divergence %: {(abs(divergence) / abs(expected_diff) * 100) if expected_diff != 0 else float('inf'):.2f}%")

    # Analyze SHAP value distribution
    shap_dim = shap_array[:, :, dim]
    print(f"    SHAP values - min: {shap_dim.min():.8f}, max: {shap_dim.max():.8f}")
    print(f"    SHAP values - mean: {shap_dim.mean():.8f}, std: {shap_dim.std():.8f}")
    print(f"    SHAP values - |values| > 0.001: {(np.abs(shap_dim) > 0.001).sum()} / {shap_dim.size}")

# Try with different background sizes
print("\n" + "=" * 80)
print("Testing with different background sizes")
print("=" * 80)

for bg_size in [1, 3, 10, 50, 100]:
    e_test = shap.DeepExplainer(model, background[:bg_size, :, :])
    sv_test = e_test.shap_values(sample_51, check_additivity=False)

    baseline_test = model(background[:bg_size, :, :]).numpy().mean(0)
    shap_test_array = sv_test[0]  # Get first sample

    print(f"\nBackground size: {bg_size}")
    for dim in range(3):
        shap_sum = shap_test_array[:, :, dim].sum()
        expected_diff = pred_sample[0, dim] - baseline_test[dim]
        divergence = shap_sum - expected_diff
        print(f"  Dim {dim}: SHAP sum={shap_sum:.6f}, Expected={expected_diff:.6f}, Div={divergence:.6f}")

print("\n" + "=" * 80)
print("Conclusion")
print("=" * 80)
print("The divergence appears to be related to:")
print("1. The specific model initialization (weights) from seed 110")
print("2. The particular input sample 51's characteristics")
print("3. Possible issues in gradient propagation through stacked LSTMs")
