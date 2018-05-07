import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os

def image_read(folder, labels):
    '''
    Reads images with filenames defined in folder using the information in labels to
    label the images extracted. Matches substrings from labels to the filename in order to succeed.
    If image dimensions does not agree, the function magically transposes it into the right dimensions. Also
    it rescales the image into 200 by 200 images, amazing!

    :param folder: Path to folder where all files are located
    :param labels: dict containing subject id and their CDR rating
    :return: _images containing list of images, _truefalse_labels containing corresponding labels
    '''

    # Start off by loading every filename in the folder
    _file_names = [folder + i for i in os.listdir(folder)]

    # Convert labels dict to list
    _subject_names = list(labels.keys())
    # Store images in this one
    _images = []
    # Store labels corresponding to index in _images in this one
    _truefalse_labels = []

    # Do for every file in our directory
    for _file in _file_names:

        # Get the subject name for this file by comparing which string in _subject_names is present in _file
        _index = [i for i, s in enumerate(_subject_names) if s in _file]
        _subject = _subject_names[_index[0]] # Cocky assumption that only one index is returned!

        # Lets do binary labels, using CDR from labels
        if labels[_subject] > 0:
            label = 1 # Alzheimer's :(
            print(_subject)
        else:
            label = 0 # No Alzheimer's! :)


        # Read image and convert to ndarray
        _image = sitk.ReadImage(_file)
        _image = sitk.GetArrayFromImage(_image)

        # Some images have wrong dimension-order, reshape!
        if _image.shape[0] < 200:
            _image = np.transpose(_image, (2, 1, 0))

        # Create a list of images
        for i in range(110, 150):
            _rescale_image = cv2.resize(_image[i, :, :], dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
            _images.append(_rescale_image)
            _truefalse_labels.append(label)

    return _images, _truefalse_labels