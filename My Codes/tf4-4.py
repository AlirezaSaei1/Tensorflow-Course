import csv
import numpy as np
import matplotlib.pyplot as plt
time_step = []
sunspots = []

# Read dataset
with open('TimeSeries/Sunspots.csv') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)
    for row in reader:
        sunspots.append(float(row[2]))
        time_step.append(int(row[0]))


# Convert lists to numpy arrays - everytime we append sth to numpy array lots of memory management for cloning the list
# so it is better to first create a throwaway list then convert it to numpy arrays
series = np.array(sunspots)
time = np.array(time_step)


# Plot the data
plt.plot(time, series)
plt.show()


# Data split
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]


# Variables
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000