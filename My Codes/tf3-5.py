# Text generation (Prediction problem)

# Import required libraries
import tensorflow as tf
import numpy as np 
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Data
data = 'Come with me to Istanbul \n  Land of turbans, spice and carpets \n This is the tale of Mr.Toot, legendary music man...'
corpus = data.lower().split('\n')


# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

total_words = len(tokenizer.word_index) + 1
print(f'word index dictionary: {tokenizer.word_index}')
print(f'total words: {total_words}')


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

# Convert the label into one-hot arrays
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Build the model
model = Sequential([
          Embedding(total_words, 64, input_length=max_seq_len-1),
          Bidirectional(LSTM(20)),
          Dense(total_words, activation='softmax')
])

# Use categorical crossentropy because this is a multi-class problem
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(inputs, ys, epochs=500)


