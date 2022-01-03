from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers.experimental.preprocessing import Rescaling

def get_model():
    model = Sequential()
    model.add(Rescaling(scale=1./255))
    model.add(Conv2D(8, 3, padding='valid'))
    model.add(MaxPooling2D(2, padding='valid'))
    model.add(Conv2D(16, 3, padding='valid'))
    model.add(MaxPooling2D(2, padding='valid'))
    model.add(Conv2D(32, 3, padding='valid'))
    model.add(MaxPooling2D(2, padding='valid'))
    model.add(Flatten())
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

