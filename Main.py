import sys
from csv_read import *
from read_images import *
from build_tfrecord import *
from DL_keras import *
from create_sets import *
from DL_DemNet import *
from DL_test1 import *
import gc
import argparse
import os

import tensorflow as tf
import keras
from keras.utils import plot_model
#from sklearn.preprocessing import MinMaxScaler


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

def display_images(images, labels, subject_name):
    '''
    Displays images in images and prints their corresponding label in labels. Only for subjects with dementia!

    :param images: List containing images
    :param labels: List containing corresponding labels
    :return: Nothing pls
    '''
    idx = 0

    for i in images:
        plt.imshow(i, cmap='gray')
        plt.title(subject_name[idx])
        plt.show()
        print(labels[idx])
        idx += 1
        plt.close()
        gc.collect()
        idx += 1



if __name__ == '__main__':

    # TODO: Go through every image visually to confirm all is well!

    # Smart-ass hardcoded file path is fixed!! Hooray!

    '''
    filename = './labels/labels.csv'
    csv_labels = read_oasis_csv(filename)

    folder = 'D:/Users/Olle Andersson/images_project_1/'
    images, labels, subject_names = image_read(folder, csv_labels)

    np.save('images_out', images)
    np.save('labels_out', labels)
    np.save('names_out', subject_names)

    print(images.shape, labels.shape)

    '''
    images = np.load('images_out.npy')
    labels = np.load('labels_out.npy')
    subject_names = np.load('names_out.npy')

    # # Check pos vs. negs
    positive_subs = [i for i in labels if i > 0]
    negative_subs = [i for i in labels if i == 0]

    x_train, y_train, x_test, y_test = split(images, labels, 0.3)

    start = 4000
    end = 4300

    #display_images(x_train[start:end, :, :], y_train[start:end], subject_names[start:end])


    x_weights = np.zeros(np.size(y_train))
    for idx, val in enumerate(y_train):
        if val > 0:
            x_weights[idx] = 3
        else:
            x_weights[idx] = 1

    positive_subs = [i for i in y_train if i > 0]
    negative_subs = [i for i in y_train if i == 0]

    print('Train data, pos = {}, neg = {}'.format(len(positive_subs), len(negative_subs)))

    positive_subs = [i for i in y_test if i > 0]
    negative_subs = [i for i in y_test if i == 0]

    print('Test data, pos = {}, neg = {}'.format(len(positive_subs), len(negative_subs)))

    # Read training images
    #train_folder = './train_images/'
    #x_train, y_train = image_read(train_folder, labels)

    # Read test images
    #test_folder = './test_images/'
    #x_test, y_test = image_read(test_folder, labels)

    # Pre-process training and test data
    x_train = x_train.reshape(x_train.shape[0], 200, 200, 1)
    x_test = x_test.reshape(x_test.shape[0], 200, 200, 1)

    # Convert to float32
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # "Normalize" between 0 - 1 TODO: Find better way, dont know if right
    #x_train /= np.max(x_train)
    #x_test /= np.max(x_test)

    # Convert labels to onehot-encoding
    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)

    #y_train = y_train.reshape(y_train.shape[0], 1, 1, 2)
    #y_test = y_test.reshape(y_test.shape[0], 1, 1, 2)

    y_train = y_train.astype('uint8')
    y_test = y_test.astype('uint8')

    # Fake call to Yupeis' function
    #model = CNN7(2, 0.5)
    model = test_net()

    plot_model(model, show_shapes=True, to_file='model.png')

    # Compile model, we use SGD and crossentropy
    model.compile(optimizer=keras.optimizers.sgd(lr=0.01), loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

    # TODO: Change validation data to something good
    history = model.fit(x = x_train, y = y_train, batch_size=32, epochs=15, verbose=1,
                        validation_data=(x_test, y_test))

    results = model.evaluate(x_test, y_test)

    print(results)
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
