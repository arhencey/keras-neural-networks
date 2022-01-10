from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np

def get_data(vocab_size, max_len):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

    # Retrieve the word index file mapping words to indices
    word_index = imdb.get_word_index()
    # Reverse the word index to obtain a dict mapping indices to words
    inverted_word_index = dict((i, word) for (word, i) in word_index.items())
    # Decode the first sequence in the dataset
    decoded_sequence = " ".join(inverted_word_index[i] for i in x_train[0])
    #print(decoded_sequence)

    #print("Number of words: ")
    #print(len(np.unique(np.hstack(x_train))))

    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)

    return x_train, y_train, x_test, y_test

#get_data(max_len=500)
