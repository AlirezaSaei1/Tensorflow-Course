# preparing features and labels

# Required libraries
import tensorflow as tf
import numpy as np


# create dataset using Dataset.range
dataset = tf.data.Dataset.range(10)
