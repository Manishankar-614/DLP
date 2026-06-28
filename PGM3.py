# PGM 3
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

model = Sequential()

for i in range(3):
    if i == 0:
        model.add(Dense(2, input_shape=(4,)))
    else:
        model.add(Dense(2))
    model.add(Activation('sigmoid'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, epochs=5, verbose=0)
print(model.evaluate(X_test, y_test))