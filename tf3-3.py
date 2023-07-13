# Embedding
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import numpy as np

# Print version of tf library
print(tf.__version__)

# Show list of datasets
tfds.list_builders()

# Load imdb_reviews datasets
imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

# 25K train sample & 25K test sample
train_data, test_data = imdb['train'], imdb['test']

# Save sentences and labels in lists - We need array of sentences for tokenizer
training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# Loop over training data and save sentences and labels
for s, l in train_data:
    training_sentences.append(s.numpy().decode('utf8'))
    training_labels.append(l.numpy())

# Loop over testing data and save sentences and labels
for s, l in test_data:
    testing_sentences.append(s.numpy().decode('utf8'))
    testing_labels.append(l.numpy())


# Convert lists to numpy arrays
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

# Made attributes variables for a clean code
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# Now that tokenizer is ready we convert is to sequences (train)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

# Apply the same to test sequences
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)