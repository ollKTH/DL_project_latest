import keras
from keras.models import Sequential
from keras.layers import *

def CNN7(_input, n_classes, _dropout):
    model = Sequential()
    #layer1
    model.add(Conv2D(96, [11, 11], strides = (4, 4),padding = 'valid' , activation = 'relu', use_bias=True, bias_initializer='zeros', input_shape = (3, 200,200)), )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #layer2
    model.add(Conv2D())
    model.add(Conv2D(256, [5, 5], strides = (2, 2),padding = 'valid' , activation = 'relu', use_bias=True, bias_initializer='zeros', input_shape = (3, 200,200)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #layer3
    model.add(Conv2D(384, [3, 3], strides = (1, 1),padding = 'valid' , activation = 'relu', use_bias=True, bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    #layer4
    model.add(Conv2D(256, [3, 3], strides = (2, 2),padding = 'valid' , activation = 'relu', use_bias=True, bias_initializer='zeros' ))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    #layer5
    model.add(Conv2D(192, [6, 6], strides = (1, 1),padding = 'valid' , activation = 'relu', use_bias=True, bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model. add(Dropout(_dropout))
    #layer6
    model.add(Conv2D(96, [1, 1], strides = (1, 1),padding = 'valid' , activation = 'relu', use_bias=True, bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(_dropout))
    #layer7
    model.add(Conv2D(2, [1, 1], strides = (2, 2),padding = 'valid' , activation = 'relu', use_bias=True, bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(_dropout))
    #output layer
    #model.add(Dense(n_classes))
    return model