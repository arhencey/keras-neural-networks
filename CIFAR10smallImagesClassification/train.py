from prepare_data import get_data
from model import get_model

X_train, y_train, X_test, y_test = get_data()
model = get_model()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test,y_test),
    epochs=250,
)


