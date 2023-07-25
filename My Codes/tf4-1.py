# Time-series

# Required libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Function used for plotting
def plot_series(time, series, format='-', start=0, end=None):
    plt.figure(figsize=(10, 6))

    if type(series) is tuple:
        for i in series:
            plt.plot(time[start:end], i[start:end], format)
    else:
        plt.plot(time[start:end], series[start:end], format)

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()


# Function to create trend
def trend(time, slope=0):
    series = slope * time
    return series


# Function to create seasonal pattern
def seasonal_pattern(season_time):
    data_pattern = np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

    return data_pattern


# Function to add seasonality
def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    data_pattern = amplitude * seasonal_pattern(season_time)
    return data_pattern


# Function to create noise
def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    noise = rnd.randn(len(time)) * noise_level
    return noise


# Function to calculate moving average
def moving_average_forecast(series, window_size):
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)


# Parameters
time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

# Update with noise
series += noise(time, noise_level, seed=42)

# Plot the results
plot_series(time, series)

# Define the split time
split_time = 1000

# Get the train set 
time_train = time[:split_time]
x_train = series[:split_time]

# Get the validation set
time_valid = time[split_time:]
x_valid = series[split_time:]


# -------------- Forecasting --------------
# Naive method
naive_forecast = series[split_time - 1:-1]

# Define time step
time_step = 100

# Print values
print(f'ground truth at time step {time_step}: {x_valid[time_step]}')
print(f'prediction at time step {time_step + 1}: {naive_forecast[time_step + 1]}')

# Plot the results
plot_series(time_valid, (x_valid, naive_forecast))