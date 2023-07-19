# Text generation (Prediction problem)

# Import required libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Data
data = 'Come with me to Istanbul \n  Land of turbans, spice and carpets \n This is the tale of Mr.Toot, legendary music man...'
corpus = data.lower().split('\n')


# Tokenizer
oov_token = '<OOV>'
tokenizer = Tokenizer(oov_token=oov_token)
tokenizer.fit_on_texts(corpus)
print(f'Length including OOV: {len(tokenizer.word_index)}')
print(tokenizer.word_index)


# Build texts
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_seq = token_list[:i+1]
        input_sequences.append(n_seq)

print(input_sequences)

# Padd sequences so they have same lengths
max_seq_len = max([len(x) for x in input_sequences])
padded_sequence = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))


# Turn padded sequences into Xs and Ys
inputs = padded_sequence[:,:-1]
labels = padded_sequence[:,-1]


