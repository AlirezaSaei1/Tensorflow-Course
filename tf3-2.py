import urllib.request, json 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


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


# Initialize the Tokenizer class
tokenizer = Tokenizer(oov_token="<OOV>")

# Generate the word index dictionary
tokenizer.fit_on_texts(sentences)

# Print the length of the word index
word_index = tokenizer.word_index
print(f'number of words in word_index: {len(word_index)}')

# Print the word index --> This has lots of words that are not necessary like: to, with, of, etc. (We might get rid of them)
print(f'word_index: {word_index}')
print()

# ---------------------------------------------------------
# Generate and pad the sequences
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')

# Print a sample headline
index = 2
print(f'sample headline: {sentences[index]}')
print(f'padded sequence: {padded[index]}')
print()

# Print dimensions of padded sequences
print(f'shape of padded sequences: {padded.shape}')
