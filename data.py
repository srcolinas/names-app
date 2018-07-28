import collections
import itertools
import logging
import os
import string
import urllib.request

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

_MEN_DATA_SOURCE = 'https://raw.githubusercontent.com/srcolinas/spanish-names/master/hombres.csv'
_WOMEN_DATA_SOURCE = 'https://raw.githubusercontent.com/srcolinas/spanish-names/master/mujeres.csv'
_CHARS_TO_INDEX = {c: i for i,c in enumerate(string.ascii_uppercase + " ÑÇ'*.")}
_VOCAB_SIZE = len(_CHARS_TO_INDEX)
_NAME_VECTOR_LENGTH = 15
_LABELS = ('Man', 'Woman')

_PARENT_PATH = os.path.dirname(os.path.realpath(__file__))
DEFAULT_FPATH = os.path.join(_PARENT_PATH, '.saved', 'data.hdf5')

def maybe_download_csv(url_csv):
    """Read the csv in the url as a string.

    Args:
        url_csv (str): the url which contains the data.

    Returns:
        A string or bytes object which containes the data from the csv
        in the url.

    """
    with urllib.request.urlopen(url_csv) as response:
        data = response.read()
    return data

def read_data_string(data, return_list=True):
    """Read the lines of the data string.

    The expected format of the string is "STRING,NUM,NUM". This function
    may be useless if the string contains a different format.

    Args:
        data (str): the string which contains the data.
        return_list (bool): whether to return the data lines as a list
            or as a generator. Defaults to True.

    Returns:
        A list which contains each line of the data contained in the
        url. If return_list is set to False, this function will return
        a generator.

    """
    # The '\r\n' separates each line of the data files. Note that we
    # ignore the header in the file
    data = (line for line in data.split(b'\r\n')[1:])
    data = (line.split(b',') for line in data)
    data = ((n.decode('utf-8'), float(f), float(a)) for n, f, a in data)
    
    if return_list:
        return list(data)
    return data

def maybe_download_and_read(url_csv, return_list=True):
    """Read the lines of the data taken from the url.

    The expected format of the .csv file contain in the url is
    "STRING,NUM,NUM". This function may be useless if the csv file
    contains a different format.

    Args:
        url_csv (str): the url which contains the data.
        return_list (bool): whether to return the data lines as a list
            or as a generator. Defaults to True.

    Returns:
        A list which contains each line of the data contained in the
        url. If return_list is set to False, this function will return
        a generator.

    """
    data = maybe_download_csv(url_csv)
    
    return read_data_string(data, return_list=return_list)

def parse_output(label):
    """Returns the label name given the label id. """
    return _LABELS[int(label)]

def parse_name(name):
    """
    Converts a name (str) into a 2d numpy array of shape
    (n_features, ). All characters should be in capital letters.

    Args:
        name (str): name in capital letters to convert.

    Returns:
        a vector representation of the name
    """
    if len(name) < _NAME_VECTOR_LENGTH:
        name = '*'*(_NAME_VECTOR_LENGTH - len(name)) + name
    
    arr = [_CHARS_TO_INDEX[name[idx]] for idx in range(_NAME_VECTOR_LENGTH)]
    return np.array(arr)

# def parse_name(name):
#     """
#     Converts a name (str) into a 1d numpy array of shape
#     (n_features, ). All characters should be in capital letters.

#     Args:
#         name (str): name in capital letters to convert.

#     Returns:
#         a vector representation of the name
#     """
#     arr = np.zeros(_OBSERVATION_SIZE)
#     for chr_, count in collections.Counter(name).items():
#         idx = _CHARS_TO_INDEX[chr_]
#         arr[idx] = count

#     return arr


def save_dataset(fpath=DEFAULT_FPATH, data_dict=None,
                return_file_object=False, **split_options):
    """ Saves the data_dict to the given path.

    This function will store the `data_dict` argument into a .hdf5 file
    using h5py to the specified fpath, which has a default value defined
    at the module level. In case `data_dict` is not provided, this will
    call the `build_dataset` function to construct the default dataset
    of this project and split it using the keywrod arguments received 
    by **split_options.

    Args:
        fpath (str): fpath in which to store the data. It defaults to 
            the module level constant DEFAULT_FPATH.
        data_dict (dict): a dictionary of the numpy arrays to store.
        return_file_object (bool): whether to return the h5py.File
            objtect created. Defaults to False.
        **split_options: keyword arguments passed to the
            `train_test_split` function from sklearn.model_selection.
            Only work if `data_dict`is None.

    """ 
    if data_dict is None:
        X, y = build_dataset()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, **split_options)
        data_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
        }

    if not fpath.endswith('hdf5'):
        fpath = fpath + '.hdf5'
    try:
        f = h5py.File(fpath, mode='w', libver='latest')
    except OSError:
        os.makedirs(os.path.dirname(fpath))
        f = h5py.File(fpath, mode='w', libver='latest')
    finally:
        for k, v in data_dict.items():
            _ = f.create_dataset(k, data=v)
        if return_file_object:
            return f
        f.close()


def load_dataset(fpath=DEFAULT_FPATH, return_arrays=True):
    """ Loads the dataset from the given fpath.

    Args:
        fpath (str): the path where the data is stored. Defaults to 
            `data.DEFAULT_FPATH`.
        return_arrays (bool): whether to return a dictionary of numpy
            arrays or the `h5py.File` object.
    returns
        A dictionary (or dictionary like object) with the data. See
        arguments for clarification.
    """

    f = h5py.File(fpath, mode='r', libver='latest')
    if return_arrays:
        data = {k:np.array(v) for k,v in f.items()}
        f.close()
        return data
    return f

def build_dataset():
    """Builds the dataset (X, y)."""

    append_fn = np.append

    lines = maybe_download_and_read(_MEN_DATA_SOURCE, return_list=False)
    men_vectors = (parse_name(name) for name, _, _ in lines)
    men_vectors = (append_fn(vector, 0) for vector in men_vectors)
        
    lines = maybe_download_and_read(_WOMEN_DATA_SOURCE, return_list=False)
    women_vectors = (parse_name(name) for name, _, _ in lines)
    women_vectors = (append_fn(vector, 1) for vector in women_vectors)

    X = np.array(list(itertools.chain(men_vectors, women_vectors)))
    np.random.shuffle(X)
    
    return X[:,:-1], X[:,-1]

def infer_input_fn(name):
    """
    Converts a name (str) into a 2d numpy array of shape
    (1, n_features). All characters should be in capital letters.

    Args:
        name (str): name in capital letters to convert.

    Returns:
        a vector representation of the name
    """
    return parse_name(name.upper())[np.newaxis, :]

if __name__ == '__main__':
    import sys
    name = ' '.join(sys.argv[1:])
    name = name.upper()
    print("{}: {}".format(name, parse_name(name)))