from prepare_data import get_data
from model import get_model

MAX_SEQ_LENGTH = 500
VOCAB_SIZE = 88585

X_train, y_train, X_test, y_test = get_data(VOCAB_SIZE, MAX_SEQ_LENGTH)
model = get_model(VOCAB_SIZE, MAX_SEQ_LENGTH)

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=5,
)
