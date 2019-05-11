import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

# Image Generator to increase variety in dataset
train_gen = ImageDataGenerator(shear_range = 0.2, zoom_range = 0.2, rescale = 1./255)
test_gen = ImageDataGenerator(rescale = 1./255)

# training data from directory, with directory names as labels
trainData = train_gen.flow_from_directory(
        directory = 'expandedData/train',
        color_mode = 'grayscale',
        target_size = (100, 100),
        class_mode = 'categorical',
        batch_size = 64
        )

# testing data from directory
testData = test_gen.flow_from_directory(
        directory = 'expandedData/test',
        color_mode = 'grayscale',
        target_size = (100, 100),
        class_mode = 'categorical',
        batch_size = 64
        )

# Creating Model - Sequential
model = Sequential()

# Adding layers to Model: Conv2D, Conv2D, MaxPooling2D, Dropout
#                         Flatten, Dense, Dropout, Dense
model.add(Conv2D(
        filters = 32,
        kernel_size=(4,4),
        strides = 1,
        activation = 'relu',
        input_shape = (100, 100, 1)
        ))
model.add(Conv2D(
        filters = 32,
        kernel_size=(4,4),
        strides = 1,
        activation = 'relu'
        ))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(
        filters = 32,
        kernel_size=(4,4),
        strides = 1,
        activation = 'relu'
        ))
model.add(Conv2D(
        filters = 32,
        kernel_size=(4,4),
        strides = 1,
        activation = 'relu'
        ))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 36, activation = 'softmax'))

# Compiling the Model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training Model to fit
model.fit_generator(trainData, steps_per_epoch = 3240//64, epochs=100)

# Saving Weights
model.save_weights('weights.h5')

# Loading Weights
model.load_weights('weights.h5')

# Prediction of full test dataset
model.predict_generator(testData, steps = 360//64, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

# Prediction of single image
import cv2
import numpy as np

chars = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
            'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
            'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

img = cv2.imread('data/mytest/test4.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

img = cv2.resize(img, (100,100))
img = np.expand_dims(img, axis=2)
img = np.expand_dims(img, axis=0)

p = list(model.predict(img)[0])
chars[p.index(1.)]