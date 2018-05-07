import sys
from csv_read import *
from read_images import *
from build_tfrecord import *
import gc
import argparse
import os

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


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

    # # Time loading the images and labelling
    folder = './images/'
    tic()
    images, other_labels = image_read(folder, labels)
    toc()
    #
    # # Check pos vs. negs
    # positive_subs = [i for i in other_labels if i > 0]
    # negative_subs = [i for i in other_labels if i == 0]
    #
    # print(len(positive_subs))
    # print(len(negative_subs))
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #
    # training = mnist.train.images
    # print(training.shape)

    folder = './tfrecordsdata/'

    # Create path to tfrecords train data
    path_tfrecords_train = os.path.join(folder, "train.tfrecords")

    # Create path to tfrecords test data
    path_tfrecords_test = os.path.join(folder, "test.tfrecords")

    # Convert to TFRecords
    convert(images = images,
            labels = other_labels,
            out_path = path_tfrecords_train)

