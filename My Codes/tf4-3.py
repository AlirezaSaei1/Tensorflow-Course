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


# Model with Simple RNN
# Default activation layer in RNNs are tanh
model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]), # 1 for univariate
    tf.keras.layers.SimpleRNN(20),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])


# Model with LSTM
# ------------------------------------------------------------------------------------------------
model_tune = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[window_size]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
])
# ------------------------------------------------------------------------------------------------

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))

# The Huber loss is the convolution of the absolute value function with the rectangular function, scaled and translated. 
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9),
              metrics=['mae'])

x_train = []
history = model.fit(x_train, epochs=19, callbacks=[lr_schedule])