# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 09:08:41 2018

@author: Aafreen Dabhoiwala
"""

# importing keras libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

#separting data into train and test from mnnist dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#converting labels into hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#train_images = train_images.astype('float32')
#test_images = test_images.astype('float32')

#reshaping
train_images =np.array(train_images).reshape(-1,28,28,1)
test_images =np.array(test_images).reshape(-1,28,28,1)


#normalizing
train_images = train_images/255.0
test_images= test_images/255.0


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution

classifier.add(Conv2D(32, (3, 3), padding = 'Same', activation="relu", input_shape=(28, 28, 1)))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#adding another convulationary layer

classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(Conv2D(64, (3, 3), activation="relu"))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 256, activation = 'relu'))
#output layer
classifier.add(Dense(output_dim = 10, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

epochs=30
batch_size=90

classifier.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs)

score = classifier.evaluate(test_images, test_labels, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# serialize model to JSON
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")

