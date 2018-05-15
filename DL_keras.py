import keras
from keras.models import Sequential
from keras.layers import *

def CNN7( n_classes, _dropout):
    model = Sequential()
    # Layer 1
    model.add(Conv2D(96, [11, 11], strides = (4, 4), activation = 'relu', padding='same', input_shape = (200, 200, 1) ))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    #model.add(BatchNormalization())
    # Layer 2
    model.add(Conv2D(256, [5, 5], strides = (2, 2), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    #model.add(BatchNormalization())
    # Layer 3
    model.add(Conv2D(384, [3, 3], strides = (1, 1), activation = 'relu'))
    #model.add(BatchNormalization())
    # Layer 4
    model.add(Conv2D(256, [3, 3], strides = (2, 2), activation = 'relu' ))
    #model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(3, 3), data_format='channels_last'))
    # Layer 5
    model.add(Conv2D(192, [6, 6], strides = (1, 1), padding='same', activation = 'relu'))
    #model.add(BatchNormalization())
    model. add(Dropout(_dropout))
    # Layer 6
    model.add(Conv2D(96, [1, 1], strides = (1, 1), padding='same', activation = 'relu'))
    #model.add(BatchNormalization())
    model.add(Dropout(_dropout))
    # Layer 7, output layer
    model.add(Conv2D(2, [1, 1], strides = (2, 2), activation = 'relu'))
    model.add(Dropout(_dropout))
    #output layer
    #model.add(Dense(n_classes))
    return model