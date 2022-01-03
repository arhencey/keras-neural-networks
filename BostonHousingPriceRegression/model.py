from keras.models import Sequential
from keras.layers import Dense

def get_model():
    model = Sequential()
    model.add(Dense(256, input_dim=13, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
