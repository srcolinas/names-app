import os
import pickle
from sklearn.tree import DecisionTreeClassifier

_PARENT_PATH = os.path.dirname(os.path.realpath(__file__))
DEFAULT_FPATH = os.path.join(_PARENT_PATH, '.saved', 'model.pkl')


def save_model(clf, fpath=DEFAULT_FPATH):

    with open(fpath, "wb") as f:
        pickle.dump(clf, f)

def load_model(fpath=DEFAULT_FPATH):

    with open(fpath, "rb") as f:
        clf = pickle.load(f)
    return clf
    
