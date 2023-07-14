# ---------------------------------------------------------
# Imports
import urllib.request, json 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
# ---------------------------------------------------------
# Variables - must try different combinations in NLP to find the best result (val_loss may increase with some combination)
vocab_size = 1000
embedding_dim = 12
max_length = 24
pad_type = 'post'
trunc_type = 'post'
oov_tok = '<OOV>'
training_size = 20000

# ---------------------------------------------------------
# Get sarcasm detection datset
with urllib.request.urlopen("https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json") as url:
    data = json.load(url)

# Initialize lists
sentences = [] 
labels = []
urls = []

# Append elements in the dictionaries into each list
for item in data:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

# ---------------------------------------------------------
# Train Test Split
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]


# ---------------------------------------------------------
# Tokenizer Part
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

# Convert training sentences to padded sequences
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=pad_type, truncating=trunc_type)

# Convert testing sentences to padded sequences
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=pad_type, truncating=trunc_type)

# ---------------------------------------------------------
# Create neural network 

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(training_padded, np.array(training_labels),
          epochs=30,
          validation_data=(testing_padded, np.array(testing_labels)),
          verbose=2)

# ---------------------------------------------------------
# Plot the results

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')