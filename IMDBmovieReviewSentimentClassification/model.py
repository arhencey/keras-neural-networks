from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.embeddings import Embedding

def get_model(num_words, seq_length):
    model = Sequential()
    model.add(Embedding(num_words, 32, input_length=seq_length))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
