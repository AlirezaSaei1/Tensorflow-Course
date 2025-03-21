# This code will be about RNNs and LSTMs
# Import required libraries
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Download the subword encoded pretokenized dataset
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

# Get the tokenizer
tokenizer = info.features['text'].encoder

BUFFER_SIZE = 10000
BATCH_SIZE = 256

# Get the train and test splits
train_data, test_data = dataset['train'], dataset['test'], 

# Shuffle the training data
train_dataset = train_data.shuffle(BUFFER_SIZE)

# Batch and pad the datasets to the maximum length of the sequences
train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_data.padded_batch(BATCH_SIZE)


# Hyperparameters
embedding_dim = 64
lstm1_dim = 64
lstm2_dim = 32
dense_dim = 64

filters = 128
kernel_size = 5


# Build the model - Model with 2 LSTMs worked better
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),

    # tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
    # tf.keras.layers.GlobalMaxPooling1D(),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm1_dim, return_sequences=True)), 
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm2_dim)),

    tf.keras.layers.Dense(dense_dim, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# --------------------------------------------------------------
# Different layers
# 1. Embedding-only (with Flatten): Fast (~5s/epoch) but overfits
# 2. Embedding with LSTM: Slower (~43s/epoch) accuracy is better but still overfits
# 3. Embedding with GRU: Faster (~20s/epoch) accuracy is good (both train and test) still overfits 
# 4. Embedding with Conv: Faster (~6s/epoch) with good accuracy but still overfits

# Over fitting has hight probability in texts becuase of OOVs
# But we can use dropouts and mixture of above models to avoid overfitting
# --------------------------------------------------------------

# Print the model summary
model.summary()

# Set the training parameters
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


NUM_EPOCHS = 10
history = model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)


# Plot utility
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

# Plot the accuracy and results 
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# NOTE: jagginess of graphs means network needs improvements