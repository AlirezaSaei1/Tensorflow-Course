import importlib
import numpy as np
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