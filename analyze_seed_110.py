"""
Detailed analysis of why seed=110 produces higher divergence.
"""

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import shap

def run_detailed_analysis(seed):
    """Run detailed analysis for a specific seed."""
    print(f"\n{'=' * 80}")
    print(f"Analyzing seed={seed}")
    print(f"{'=' * 80}")

    # Clear session
    tf.keras.backend.clear_session()

    # Set all seeds
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(seed)

    # Use modern numpy generator
    rng = np.random.default_rng(seed)

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
    df_valid = df.iloc[train_split:valid_split]
    df_test = df.iloc[valid_split:]

    n_features = df.shape[1]
    train_dataset = windowed_dataset(series=df_train.values, in_horizon=100, out_horizon=3, delay=1, batch_size=32)

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

    # Create explainer
    e = shap.DeepExplainer(model, xarr[:100, :, :])
    sv = e.shap_values(xarr[:100, :, :], check_additivity=False)
    model_output_values = model(xarr[:100, :, :])

    print("\nModel weight statistics:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
            weights = layer.get_weights()
            print(f"  Layer {i} ({layer.name}):")
            for j, w in enumerate(weights):
                print(f"    Weight {j}: shape={w.shape}, mean={w.mean():.6f}, std={w.std():.6f}, min={w.min():.6f}, max={w.max():.6f}")

    print("\nData statistics:")
    print(f"  Input data: mean={xarr[:100].mean():.6f}, std={xarr[:100].std():.6f}, min={xarr[:100].min():.6f}, max={xarr[:100].max():.6f}")

    print("\nModel output statistics:")
    for dim in range(3):
        output_dim = model_output_values[:, dim].numpy()
        print(f"  Dimension {dim}: mean={output_dim.mean():.6f}, std={output_dim.std():.6f}, min={output_dim.min():.6f}, max={output_dim.max():.6f}")

    print("\nExpected value statistics:")
    for dim in range(3):
        print(f"  Dimension {dim}: {e.expected_value[dim].numpy():.6f}")

    print("\nDivergence analysis:")
    all_diffs = []
    for dim in range(3):
        diff = (
            model_output_values[:, dim].numpy()
            - e.expected_value[dim].numpy()
            - sv[dim].sum(axis=tuple(range(1, sv[dim].ndim)))
        )
        max_diff = diff.max()
        min_diff = diff.min()
        mean_diff = diff.mean()
        std_diff = diff.std()

        print(f"\n  Dimension {dim}:")
        print(f"    Max divergence:  {max_diff:.8f}")
        print(f"    Min divergence:  {min_diff:.8f}")
        print(f"    Mean divergence: {mean_diff:.8f}")
        print(f"    Std divergence:  {std_diff:.8f}")
        print(f"    Samples with |div| > 0.05: {(np.abs(diff) > 0.05).sum()}")

        # Find the sample with max divergence
        max_idx = np.abs(diff).argmax()
        print(f"    Sample with max divergence: index={max_idx}")
        print(f"      Model output: {model_output_values[max_idx, dim].numpy():.6f}")
        print(f"      Expected value: {e.expected_value[dim].numpy():.6f}")
        print(f"      SHAP sum: {sv[dim][max_idx].sum():.6f}")
        print(f"      Divergence: {diff[max_idx]:.6f}")

        all_diffs.append(diff)

    overall_max = max(np.abs(d).max() for d in all_diffs)
    print(f"\n  Overall max divergence: {overall_max:.8f}")

    return overall_max

# Analyze seed 110 (the problematic one)
div_110 = run_detailed_analysis(110)

# Compare with a few good seeds
print("\n\n" + "=" * 80)
print("COMPARISON WITH OTHER SEEDS")
print("=" * 80)

good_seeds = [42, 0, 1, 2, 3]
results = [("110 (problematic)", div_110)]

for seed in good_seeds:
    div = run_detailed_analysis(seed)
    results.append((f"{seed}", div))

print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

for name, div in results:
    status = "❌ FAIL" if div >= 0.05 else "✅ PASS"
    print(f"  Seed {name:20s}: max divergence = {div:.8f}  {status}")

print(f"\nRecommended tolerance: {max(r[1] for r in results) * 1.1:.8f}")
