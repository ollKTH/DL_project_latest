import csv, sys

def read_oasis_csv(filename):
    '''
    A function that reads amazing .csv files and converts them to magic dicts.

    :param filename: The filename of a .csv file to read
    :return: _labels containing subject name and their corresponding CDR
    '''

    _labels = {}

    with open(filename, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                _number = float(row[1])
                _labels[row[0]] = float(_number)
            except:
                print('Error!')

    return _labels