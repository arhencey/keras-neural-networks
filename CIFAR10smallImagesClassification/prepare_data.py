from keras.datasets import cifar10
import numpy as np

from PIL import Image as im
from matplotlib import pyplot as plt

def get_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # convert ground truths to one-hot
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    y_train_one_hot = np.zeros((y_train.size, y_train.max()+1))
    y_train_one_hot[np.arange(y_train.size),y_train] = 1
    y_test_one_hot = np.zeros((y_test.size, y_test.max()+1))
    y_test_one_hot[np.arange(y_test.size),y_test] = 1
    return X_train, y_train_one_hot, X_test, y_test_one_hot
