"""
Run the stacked LSTM test multiple times to determine actual divergence values.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import shap

print("=" * 80)
print("Testing stacked LSTM divergence across multiple runs")
print("=" * 80)

def run_stacked_lstm_test(run_number):
    """Run the stacked LSTM test and return divergence values."""
    print(f"\n{'=' * 80}")
    print(f"Run #{run_number}")
    print(f"{'=' * 80}")

    # Define the start and end datetime
    start_datetime = pd.to_datetime("2020-01-01 00:00:00")
    end_datetime = pd.to_datetime("2023-03-31 23:00:00")

    # Generate a DatetimeIndex with hourly frequency
    date_rng = pd.date_range(start=start_datetime, end=end_datetime, freq="h")

    # Create a DataFrame with random data for 7 features
    num_samples = len(date_rng)
    num_features = 7

    # Generate random data for the DataFrame
    data = np.random.rand(num_samples, num_features)

    # Create the DataFrame with a DatetimeIndex
    df = pd.DataFrame(data, index=date_rng, columns=[f"X{i}" for i in range(1, num_features + 1)])

    def windowed_dataset(series=None, in_horizon=None, out_horizon=None, delay=None, batch_size=None):
        total_horizon = in_horizon + out_horizon
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(total_horizon, shift=delay, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(total_horizon))
        dataset = dataset.map(lambda window: (window[:-out_horizon, :], window[-out_horizon:, 0]))
        dataset = dataset.batch(batch_size).prefetch(1)
        return dataset

    # Define the proportions for the splits
    train_size = 0.4
    valid_size = 0.5

    # Calculate the split points
    train_split = int(len(df) * train_size)
    valid_split = int(len(df) * (train_size + valid_size))

    # Split the DataFrame
    df_train = df.iloc[:train_split]
    df_valid = df.iloc[train_split:valid_split]
    df_test = df.iloc[valid_split:]

    # number of input features and output targets
    n_features = df.shape[1]

    # split the data into sliding sequential windows
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

    # Create an explainer object
    e = shap.DeepExplainer(model, xarr[:100, :, :])

    # Calculate SHAP values for the data
    sv = e.shap_values(xarr[:100, :, :], check_additivity=False)
    model_output_values = model(xarr[:100, :, :])

    # Check divergence for each output dimension
    divergences = []
    for dim in range(3):
        div = (
            model_output_values[:, dim].numpy()
            - e.expected_value[dim].numpy()
            - sv[dim].sum(axis=tuple(range(1, sv[dim].ndim)))
        )
        max_div = np.abs(div).max()
        mean_div = np.abs(div).mean()
        divergences.append({
            'dim': dim,
            'max': max_div,
            'mean': mean_div,
            'all_values': div
        })

        print(f"  Dimension {dim}:")
        print(f"    Max divergence:  {max_div:.8f}")
        print(f"    Mean divergence: {mean_div:.8f}")
        print(f"    Min value: {div.min():.8f}")
        print(f"    Max value: {div.max():.8f}")

    overall_max = max(d['max'] for d in divergences)
    overall_mean = np.mean([d['mean'] for d in divergences])

    print(f"\n  Overall:")
    print(f"    Max divergence across all dimensions:  {overall_max:.8f}")
    print(f"    Mean divergence across all dimensions: {overall_mean:.8f}")

    return divergences, overall_max, overall_mean

# Run the test 3 times
all_results = []
for run in range(1, 4):
    try:
        divergences, overall_max, overall_mean = run_stacked_lstm_test(run)
        all_results.append({
            'run': run,
            'overall_max': overall_max,
            'overall_mean': overall_mean,
            'divergences': divergences
        })
    except Exception as e:
        print(f"\n✗ Run #{run} failed with error: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("SUMMARY ACROSS ALL RUNS")
print("=" * 80)

if all_results:
    max_divergences = [r['overall_max'] for r in all_results]
    mean_divergences = [r['overall_mean'] for r in all_results]

    print(f"\nMax divergences per run:")
    for i, max_div in enumerate(max_divergences, 1):
        print(f"  Run {i}: {max_div:.8f}")

    print(f"\nMean divergences per run:")
    for i, mean_div in enumerate(mean_divergences, 1):
        print(f"  Run {i}: {mean_div:.8f}")

    print(f"\nStatistics:")
    print(f"  Highest max divergence:  {max(max_divergences):.8f}")
    print(f"  Lowest max divergence:   {min(max_divergences):.8f}")
    print(f"  Average max divergence:  {np.mean(max_divergences):.8f}")
    print(f"  Std dev max divergence:  {np.std(max_divergences):.8f}")

    print(f"\nRecommended tolerance:")
    # Add some margin (e.g., 2x the max observed)
    recommended = max(max_divergences) * 1.5
    print(f"  Based on observed values: {recommended:.8f}")
    print(f"  Rounded up for safety: {np.ceil(recommended * 100) / 100:.2f}")

    # Check current tolerance
    current_tolerance = 0.05
    print(f"\nCurrent test tolerance: {current_tolerance}")
    if max(max_divergences) <= current_tolerance:
        print("  ✓ Current tolerance is sufficient!")
    else:
        print(f"  ✗ Current tolerance is too strict!")
        print(f"  → Need to increase to at least {np.ceil(max(max_divergences) * 100) / 100:.2f}")
else:
    print("\n✗ No successful runs!")
