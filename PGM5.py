# PGM 5
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

train_X = train_X.reshape(-1, 28, 28, 1) / 255
test_X = test_X.reshape(-1, 28, 28, 1) / 255
train_Y = to_categorical(train_Y)
test_Y = to_categorical(test_Y)

model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_X, train_Y, epochs=10, batch_size=64, verbose=0)

loss, acc = model.evaluate(test_X, test_Y, verbose=0)
print("Test loss", loss)
print("Test accuracy", acc)

pred = np.argmax(model.predict(test_X, verbose=0)[0])
print(pred)

plt.imshow(test_X[0].reshape(28,28), cmap='gray')
plt.show()