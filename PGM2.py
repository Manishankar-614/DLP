# PGM 2
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

X, y = load_iris(return_X_y=True)
y = to_categorical(y, 3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

def run_model(act):
    print(f"\n=== {act.upper()} ===")

    model = Sequential([
        Dense(4, activation=act, input_shape=(4,)),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary() 
    model.fit(X_train, y_train, epochs=5, verbose=0)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss: {loss:.3f}")
    print(f"Accuracy: {acc:.3f}")

for act in ['sigmoid', 'relu', 'tanh']:
    run_model(act)