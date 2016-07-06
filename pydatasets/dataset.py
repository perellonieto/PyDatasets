__docformat__ = 'restructedtext en'
import warnings
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import StratifiedKFold
import numpy as np

__author__ = "Miquel Perello Nieto"
__credits__ = ["Miquel Perello Nieto"]

__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Miquel Perello Nieto"
__email__ = "miquel@perellonieto.com"
__status__ = "Development"

import urllib2
from os.path import isfile

class Dataset(object):
    def __init__(self, name, data, target):
        self.name = name
        self._data = self.standardize_data(data)
        self._target, self._classes, self._names, self._counts = self.standardize_targets(target)

    def standardize_data(self, data):
        new_data = data.astype(float)
        data_mean = new_data.mean(axis=0)
        data_std = new_data.std(axis=0)
        data_std[data_std == 0] = 1
        return (new_data-data_mean)/data_std

    def standardize_targets(self, target):
        target = np.squeeze(target)
        names, counts = np.unique(target, return_counts=True)
        new_target = np.empty_like(target, dtype=int)
        for i, name in enumerate(names):
            new_target[target==name] = i
        classes = range(len(names))
        return new_target, classes, names, counts

    def separate_sets(self, x, y, test_fold_id, test_folds):
        x_test = x[test_folds == test_fold_id, :]
        y_test = y[test_folds == test_fold_id]

        x_train = x[test_folds != test_fold_id, :]
        y_train = y[test_folds != test_fold_id]
        return [x_train, y_train, x_test, y_test]

    def reduce_number_instances(self, proportion=0.1):
        skf = StratifiedKFold(self._target, n_folds=1.0/proportion)
        test_folds = skf.test_folds
        _, _, self._data, self._target = self.separate_sets(
                                    self._data, self._target, 0, test_folds)

    @property
    def target(self):
        return self._target

    #@target.setter
    #def target(self, new_value):
    #    self._target = new_value

    @property
    def data(self):
        return self._data

    @property
    def names(self):
        return self._names

    @property
    def classes(self):
        return self._classes

    @property
    def counts(self):
        return self._counts

    def print_summary(self):
        print self

    @property
    def n_classes(self):
        return len(self._classes)

    def __str__(self):
        return("Name = {}\n"
               "Data shape = {}\n"
               "Target shape = {}\n"
               "Target classes = {}\n"
               "Target labels = {}").format(self.name, self.data.shape,
                                            self.target.shape, self.classes,
                                            self.names)
