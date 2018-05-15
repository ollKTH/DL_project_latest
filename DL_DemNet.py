import keras
from keras.models import Sequential
from keras.layers import *

def DemNet(_dropout):
    model = Sequential()

    model.add(Conv2D(96, [11, 11], strides=(4, 4), activation='relu', padding='same', input_shape=(200, 200, 1)))

    # Layer 1
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid'))

    # Layer 2
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid'))

    # Layer 3
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_last'))

    # Dropout
    model.add(Dropout(_dropout))

    # Layer 4
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='valid'))

    # Layer 5
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='valid'))

    # Layer 6
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last'))

    # Dropout
    model.add(Dropout(_dropout))

    # Layer 7
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='valid'))

    # Layer 8
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='valid'))

    # Layer 9
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='valid'))

    # Layer 10
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last'))

    # Dropout
    model.add(Dropout(_dropout))

    # Layer 7
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='valid'))

    # Layer 8
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='valid'))

    # Layer 9
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='valid'))

    # Layer 10
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last'))

    # Dropout
    model.add(Dropout(_dropout))

    # Layer 7
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='valid'))

    # Layer 8
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='valid'))

    # Layer 9
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='valid'))

    # Layer 10
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last'))

    # Dropout
    model.add(Dropout(_dropout))

    model.add(Dense(256))

    model.add(Dropout(_dropout))

    model.add(Dense(256))

    model.add(Dropout(_dropout))

    model.add(Dense(2))

