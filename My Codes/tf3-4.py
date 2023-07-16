# This code will be about RNNs and LSTMs
import tensorflow as tf

# ....

# tokenizer.vocab_size
vocab_size = 0

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# ....