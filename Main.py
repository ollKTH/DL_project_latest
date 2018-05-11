import sys
from csv_read import *
from read_images import *
from build_tfrecord import *
import gc
import argparse
import os

import tensorflow as tf
import keras


def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

def display_images(images, labels):
    '''
    Displays images in images and prints their corresponding label in labels. Only for subjects with dementia!

    :param images: List containing images
    :param labels: List containing corresponding labels
    :return: Nothing pls
    '''
    idx = 0

    for i in images:
        if labels[idx] > 0:
            plt.imshow(i, cmap='gray')
            plt.show()
            print(labels[idx])
            idx += 1
            plt.close()
            gc.collect()
        idx += 1



if __name__ == '__main__':

    # TODO: Go through every image visually to confirm all is well!

    # Smart-ass hardcoded file path is fixed!! Hooray!
    filename = './labels/labels.csv'
    labels = read_oasis_csv(filename)

    # Read training images
    train_folder = './train_images/'
    x_train, y_train = image_read(train_folder, labels)

    # Read test images
    test_folder = './test_images/'
    x_test, y_test = image_read(test_folder, labels)

    print(np.max(x_test))

    # Pre-process training and test data
    dimData = np.prod(x_train.shape[1:])
    x_train = x_train.reshape(x_train.shape[0], dimData)
    x_test = x_test.reshape(x_test.shape[0], dimData)

    # Convert to float32
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # "Normalize" between 0 - 1 TODO: Find better way, dont know if right
    x_train /= np.max(x_train)
    x_test /= np.max(x_test)

    # Convert labels to onehot-encoding
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    # Fake call to Yupeis' function
    model = DL_keras.CNN7()

    # Compile model, we use SGD and crossentropy
    model.compile(optimizer=keras.optimizers.sgd(lr='0.01'), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    # TODO: Change validation data to something good
    results = model.fit(x_train, y_train, batch_size=256, epochs=20, verbose=1,
                   validation_data=(x_test, y_test))

    # TODO: Add plot for accuracy over time!