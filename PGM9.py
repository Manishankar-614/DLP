# PGM 9
from numpy import expand_dims
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

img = image.load_img('cat.png')

data = image.img_to_array(img)

samples = expand_dims(data, 0)

datagen = ImageDataGenerator(width_shift_range=0.2)

it = datagen.flow(samples, batch_size=1)

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    batch = next(it)
    pyplot.imshow(batch[0].astype('uint8'))

pyplot.show()