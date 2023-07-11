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
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>') # OOV == Out of Vocabulary
tokenizer.fit_on_texts(sentences)
word_idx = tokenizer.word_index
print(f'Word Indicies:\n{word_idx}')

sequences = tokenizer.texts_to_sequences(sentences)
print(f'Text to sequence:\n{sequences}')


# Test sentences
test_sentences = [
    'I really love my dog',
    'My dog loves your cat!'
]

test_sequences = tokenizer.texts_to_sequences(test_sentences)
print(f'Test text to sequence:\n{test_sequences}')

converted_sequences_to_text = tokenizer.sequences_to_texts(test_sequences)
print(f'Test sequence to text:\n{converted_sequences_to_text}')
