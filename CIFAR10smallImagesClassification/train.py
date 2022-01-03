from prepare_data import get_data
from model import get_model
from tensorflow.keras.callbacks import EarlyStopping

X_train, y_train, X_test, y_test = get_data()
model = get_model()

es = EarlyStopping(monitor='val_loss', mode='min', patience=5)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test,y_test),
    epochs=250,
    callbacks=[es]
)


