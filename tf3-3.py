# Embedding
import tensorflow as tf
print(tf.__version__)

import tensorflow_datasets as tfds
tfds.list_builders()
imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

import numpy as np
train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_lables = []

testing_sentences = []
testing_lables = []

# Loop over training data and save sentences and labels
for s, l in train_data:
    training_sentences.append(s.numpy().decode('utf8'))
    training_lables.append(l.numpy())

# Loop over testing data and save sentences and labels
for s, l in test_data:
    testing_sentences.append(s.numpy().decode('utf8'))
    testing_lables.append(l.numpy())