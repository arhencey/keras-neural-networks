from keras.datasets import mnist
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    #(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    #plt.imshow(x_train[0], cmap='gray_r')
    #plt.show()

    # expand to 3d, e.g. add channels
    X = np.expand_dims(x_train, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return [X, y_train]
