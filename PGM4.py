# PPGM 4
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i])
    plt.title(y_train[i])
    plt.axis('off')
plt.show()

X_train = X_train.reshape(-1, 784) / 255
X_test = X_test.reshape(-1, 784) / 255

print("New shape of X_train:", X_train.shape)
print("New shape of X_test:", X_test.shape)

model = Sequential([
    Input(shape=(784,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=10, verbose=0)

train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

print("\nModel Summary")
model.summary()

print("\nTraining Data Evaluation")
print(classification_report(y_train, train_pred))

print("\nTest Data Evaluation")
print(classification_report(y_test, test_pred))