# This code will be about RNNs and LSTMs
import tensorflow as tf

# ....

# tokenizer.vocab_size
vocab_size = 64

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64),
    # LSTMs can be stacked like Dense layers
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    # Desired output is 64 but becuase it is bidirectional in model.sumamry it has 128 output
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# ....