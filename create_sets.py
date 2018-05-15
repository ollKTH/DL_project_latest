import numpy as np

def split(_images, _labels, _factor):
    '''
    Splits _images and _labels into training data and test data

    :param _images:
    :param _labels:
    :param _factor:
    :return:
    '''

    # First just try splitting straight off
    _length = len(_labels)
    _split_idx = np.round(_length * (1 - _factor))

    _split_idx = np.int32(_split_idx)

    _x_train = _images[1:_split_idx, :, :]
    _y_train = _labels[1:_split_idx]

    _x_test = _images[_split_idx+1:, :, :]
    _y_test = _labels[_split_idx+1:]

    return _x_train, _y_train, _x_test, _y_test



