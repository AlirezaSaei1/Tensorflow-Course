# preparing features and labels

# Required libraries
import tensorflow as tf
import numpy as np


# create dataset using Dataset.range
dataset = tf.data.Dataset.range(10)
for val in dataset:
    print(val.numpy())


# Create windows without fixed size
windows = dataset.window(5, shift=1)

# Create widnows with fixed size
windows = dataset.window(5, shift=1, drop_remainder=True)
for window in windows:
    for val in window:
        print(val.numpy(), end=" ")
    print('')


# Flatten
flat = windows.flat_map(lambda window: window.batch(5))

# SPlit data into features and labels
splitted = flat.map(lambda window: (window[:-1], window[-1]))
for x, y in splitted:
    print(x.numpy(), y.numpy())