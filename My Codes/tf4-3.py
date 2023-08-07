import importlib
import numpy as np
import tensorflow as tf

tf4_1 = importlib.import_module("tf4-1")

# Parameters
time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = baseline + tf4_1.trend(time, slope) + tf4_1.seasonality(time, period=365, amplitude=amplitude)

# Update with noise
series += tf4_1.noise(time, noise_level, seed=42)

# Plot the results
tf4_1.plot_series(time, series)


# Define the split time
split_time = 1000

# Get the train set 
time_train = time[:split_time]
x_train = series[:split_time]

# Get the validation set
time_valid = time[split_time:]
x_valid = series[split_time:]

# Parameters
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

dataset = tf4_1.windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

