import os
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

_PARENT_PATH = os.path.dirname(os.path.realpath(__file__))
DEFAULT_FPATH = os.path.join(_PARENT_PATH, 'saved', 'model.pkl')

REGISTRY = {
    'tree': DecisionTreeClassifier, 'forest': RandomForestClassifier
}

def save_model(clf, fpath=DEFAULT_FPATH):
    """Saves the classifier in the given file path.

    Args:
        clf (obj): A classifier object such as
            `sklearn.tree.DecisionTreeClassifier`.
        fpath (str): the file path to store the classifier. Defaults to
            `model.DEFAULT_FPATH`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(clf, f)

def load_model(fpath=DEFAULT_FPATH):
    """Loads the classifier from the given file path.

    Args:
        fpath (str): the file path with the serializede classifier.
            Defaults to`model.DEFAULT_FPATH`.
    
    Returns:
        A classifier object such as `sklearn.tree.DecisionTreeClassifier`.
    """
    with open(fpath, "rb") as f:
        clf = pickle.load(f)
    return clf
    
