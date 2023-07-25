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
def noise(time, noise_level=1, seed=None);
    rnd = np.random.RandomState(seed)
    noise = rnd.randn(len(time)) * noise_level
    return noise
