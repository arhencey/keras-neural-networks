from keras.datasets import boston_housing
from sklearn.preprocessing import MinMaxScaler

# Loads the Boston Housing price regression dataset
def get_data():
    train, test = boston_housing.load_data()

    X_train = train[0]
    y_train = train[1]
    X_test = test[0]
    y_test = test[1]

    # Normalize the features
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test
