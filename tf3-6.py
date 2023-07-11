import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat',
    # Punctuation marks are stripped
    'You love my dog!',
    # Sentence with different length
    'Do you think my dog is amazing?'
]

# When you don't know the number of words this will take the most frequent ones
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_idx = tokenizer.word_index
print(word_idx)

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)