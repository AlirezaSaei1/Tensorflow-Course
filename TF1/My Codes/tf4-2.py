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
dataset = dataset.window(5, shift=1, drop_remainder=True)
for window in dataset:
    for val in window:
        print(val.numpy(), end=" ")
    print('')


# Flatten
dataset = dataset.flat_map(lambda window: window.batch(5))

# SPlit data into features and labels
dataset = dataset.map(lambda window: (window[:-1], window[-1]))

# Shuffle data 
dataset_shuffled = dataset.shuffle(buffer_size=10)
for x, y in dataset_shuffled:
    print(x.numpy(), y.numpy())

# Create batches of 2
batch = dataset_shuffled.batch(2).prefetch(1)
for x, y in batch:
    print(f'x={x.numpy()}')
    print(f'y={y.numpy()}')