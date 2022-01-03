from prepare_data import get_data
from model import get_model

X_train, y_train, X_test, y_test = get_data()
model = get_model()

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=250,
    )

# Evaluate the model on the test data
print("Evaluate on test data")
results = model.evaluate(X_test, y_test)
print("test mean absolute error:", results[1])
