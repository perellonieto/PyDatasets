__docformat__ = 'restructedtext en'
import warnings
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import StratifiedKFold
import numpy as np

from dataset import Dataset
from data import Data

__author__ = "Miquel Perello Nieto"
__credits__ = ["Miquel Perello Nieto"]

__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Miquel Perello Nieto"
__email__ = "miquel@perellonieto.com"
__status__ = "Development"

import urllib2
from os.path import isfile

def test_datasets(dataset_names):
    from sklearn.svm import SVC
    from sklearn.cross_validation import StratifiedKFold
    data = Data(dataset_names=dataset_names)

    def separate_sets(x, y, test_fold_id, test_folds):
        x_test = x[test_folds == test_fold_id, :]
        y_test = y[test_folds == test_fold_id]

        x_train = x[test_folds != test_fold_id, :]
        y_train = y[test_folds != test_fold_id]
        return [x_train, y_train, x_test, y_test]

    n_folds = 2
    accuracies = {}
    for name, dataset in data.datasets.iteritems():
        dataset.print_summary()
        skf = StratifiedKFold(dataset.target, n_folds=n_folds, shuffle=True)
        test_folds = skf.test_folds
        accuracies[name] = np.zeros(n_folds)
        for test_fold in np.arange(n_folds):
            x_train, y_train, x_test, y_test = separate_sets(
                    dataset.data, dataset.target, test_fold, test_folds)

            svc = SVC(C=1.0, kernel='rbf', degree=1, tol=0.01)
            svc.fit(x_train, y_train)
            prediction = svc.predict(x_test)
            accuracies[name][test_fold] = 100*np.mean((prediction == y_test))
            print("Acc = {0:.2f}%".format(accuracies[name][test_fold]))
    return accuracies

def test():
    datasets_li2014 = ['abalone', 'balance-scale', 'credit-approval',
            'dermatology', 'ecoli', 'german', 'heart-statlog', 'hepatitis',
            'horse', 'ionosphere', 'lung-cancer', 'libras-movement',
            'mushroom', 'diabetes', 'landsat-satellite', 'segment',
            'spambase', 'wdbc', 'wpbc', 'yeast']

    datasets_hempstalk2008 = ['diabetes', 'ecoli', 'glass',
            'heart-statlog', 'ionosphere', 'iris', 'letter',
            'mfeat-karhunen', 'mfeat-morphological', 'mfeat-zernike',
            'optdigits', 'pendigits', 'sonar', 'vehicle', 'waveform-5000']

    datasets_others = [ 'diabetes', 'ecoli', 'glass', 'heart-statlog',
            'ionosphere', 'iris', 'letter', 'mfeat-karhunen',
            'mfeat-morphological', 'mfeat-zernike', 'optdigits',
            'pendigits', 'sonar', 'vehicle', 'waveform-5000',
            'scene-classification', 'tic-tac', 'autos', 'car', 'cleveland',
            'dermatology', 'flare', 'page-blocks', 'segment', 'shuttle',
            'vowel', 'zoo', 'abalone', 'balance-scale', 'credit-approval',
            'german', 'hepatitis', 'lung-cancer']

    # Datasets that we can add but need to be reduced
    datasets_to_add = ['MNIST']

    dataset_names = list(set(datasets_li2014 + datasets_hempstalk2008 +
        datasets_others))

    accuracies = test_datasets(dataset_names)
    for i, name in enumerate(dataset_names):
        if name in accuracies.keys():
            print("{}. {} Acc = {:.2f}% +- {:.2f}".format(
                  i+1, name, accuracies[name].mean(), accuracies[name].std()))
        else:
            print("{}. {}  Not Available yet".format(i+1, name))

if __name__=='__main__':
    test()
