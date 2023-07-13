__docformat__ = 'restructedtext en'
import warnings
import json
#import openml
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle as skl_shuffle
import numpy as np

from pydataset import data as pydataset_data

import urllib
from os.path import isfile


#from .synthetic.fruits import get_fruits

datasets_binary = ['credit-approval', 'diabetes',
            'german', 'heart-statlog', 'hepatitis',
            'horse', 'ionosphere', 'lung-cancer',
            'mushroom', 'scene-classification',
            'sonar', 'spambase', 'tic-tac',
            'wdbc', 'wpbc']

datasets_li2014 = ['abalone', 'balance-scale', 'credit-approval',
            'dermatology', 'german', 'heart-statlog', 'hepatitis',
            'horse', 'ionosphere', 'lung-cancer', 'libras-movement',
            'mushroom', 'diabetes', 'landsat-satellite', 'segment',
            'spambase', 'wdbc', 'wpbc']

datasets_hempstalk2008 = ['diabetes',
        'heart-statlog', 'ionosphere', 'iris', 'letter',
        'mfeat-karhunen', 'mfeat-morphological', 'mfeat-zernike',
        'optdigits', 'pendigits', 'sonar', 'vehicle', 'waveform-5000']

datasets_others = [ 'diabetes', 'heart-statlog',
        'ionosphere', 'iris', 'letter', 'mfeat-karhunen',
        'mfeat-morphological', 'mfeat-zernike', 'optdigits',
        'pendigits', 'sonar', 'vehicle', 'waveform-5000',
        'scene-classification', 'tic-tac', 'autos', 'car', 'cleveland',
        'dermatology', 'flare', 'page-blocks', 'segment', 'shuttle',
        'vowel', 'abalone', 'balance-scale', 'credit-approval',
        'german', 'hepatitis', 'lung-cancer', 'ecoli', 'glass', 'yeast', 'zoo']

datasets_big = ['abalone', 'car', 'flare', 'german', 'landsat-satellite',
                'letter', 'mfeat-karhunen', 'mfeat-morphological',
                'mfeat-zernike', 'mushroom', 'optdigits', 'page-blocks',
                'pendigits', 'scene-classification', 'segment', 'shuttle',
                'spambase', 'waveform-5000', 'yeast']

datasets_small_example = ['iris', 'spambase', 'autos']

datasets_multilabel = ['birds', 'enron', 'emotions', 'fruits', 'scene',
                       'yeast', 'reuters', 'genbase', 'slashdot', 'image',
                       'sklearn-multilabel', 'mlgaussians']

datasets_all = list(set(datasets_li2014 + datasets_hempstalk2008 +
                        datasets_others + datasets_binary))

datasets_non_binary = [d for d in datasets_all if d not in datasets_binary]

class Dataset(object):
    def __init__(self, name, data, target, feature_names=None,
                 shuffle=False, random_state=None):
        '''
        Parameters
        ==========
        data: numpy array or DataFrame (n_samples, n_features)
            If it is a DataFrame, the feature names are extracted from the
            column names.
        target: numpy array or DataFrame (n_samples, n_classes) or (n_samples, 1)
            if size is (n_samples, n_classes) can be multiclass or multi-label
            if size is (n_samples, 1) is assumed to be multiclass problem
            If it is a DataFrame with multiple columns, the target names is
            extracted from the column names.
        '''
        self.name = name

        if feature_names is None:
            if hasattr(data, 'columns'):
                feature_names = data.columns.values
            else:
                feature_names = range(len(data[0]))

        self.feature_names = feature_names

        self._data = self.standardize_data(data)
        self._target, self._classes, self._names, self._counts = self.standardize_targets(target)
        if shuffle:
            self.shuffle(random_state=random_state)

    def shuffle(self, random_state=None):
        self._data, self._target = skl_shuffle(self._data, self._target,
                                               random_state=random_state)

    def standardize_data(self, data):
        new_data = data.astype(float)
        data_mean = new_data.mean(axis=0)
        data_std = new_data.std(axis=0)
        data_std[data_std == 0] = 1
        return (new_data-data_mean)/data_std

    def standardize_targets(self, target):
        '''

        target: numpy array or DataFrame (n_samples, 1) or (n_samples, n_classes)
            If it is a DataFrame with multiple columns, the target names are
            obtained from the column names.
            If there is only one column, the target names are obtained from the
            values in each row.
        '''
        target = np.squeeze(target)
        if ((len(target.shape) > 1)
            and (target.shape[1] > 1)):
            if hasattr(target, 'columns'):
                names = target.columns
                counts = np.sum(target, axis=0).values
                new_target = target.astype(int).values
            else:
                names = list(range(target.shape[1]))
                counts = np.sum(target, axis=0)
                new_target = target.astype(int)
        else:
            names, counts = np.unique(target, return_counts=True)
            new_target = np.zeros_like(target, dtype=int)
            for i, name in enumerate(names):
                new_target[target==name] = i
        classes = list(range(len(names)))
        if type(names[0]) is np.ndarray:
            names = [''.join(name) for name in names]
        else:
            names = [str(name) for name in names]
        return new_target, classes, names, counts

    def separate_sets(self, x, y, test_fold_id, test_folds):
        x_test = x[test_folds == test_fold_id, :]
        y_test = y[test_folds == test_fold_id]

        x_train = x[test_folds != test_fold_id, :]
        y_train = y[test_folds != test_fold_id]
        return [x_train, y_train, x_test, y_test]

    def reduce_number_instances(self, proportion=0.1):
        skf = StratifiedKFold(n_splits=int(1.0/proportion))
        test_folds = skf.test_folds
        train_idx, test_idx = next(iter(skf.split(X=self._data,
                                                  y=self._target)))
        self._data, self._target = self._data[test_idx], self._target[test_idx]

    @property
    def is_semisupervised(self):
        '''
        Indicates if the target of any instance does not belong to at least one
        class. Assumes that a target of the shape (n_samples, 1) is always
        fully supervised, while a target of shape (n_samples, n_classes) can be
        unsupervised if the full row contains zeros.
        '''
        if (len(self._target.shape) > 1)  and (self._target.shape[1] > 1):
            return np.any(np.sum(self._target, axis=1) == 0)
        return False

    @property
    def is_multilabel(self):
        '''
        Indicates if the target of any instance has more than one class
        assignation. If the target is of the shape (n_samples, 1) it is assumed
        not to be a multilabel problem. If the target is of the shape
        (n_samples, n_classes) it is considered multilabel if at least one of
        the rows contains more than one one.
        '''
        if (len(self._target.shape) > 1) and (self._target.shape[1] > 1):
            return np.any(np.sum(self._target, axis=1) > 1)
        return False

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
        print(self)

    @property
    def n_classes(self):
        return len(self._classes)

    @property
    def n_features(self):
        return self._data.shape[1]

    @property
    def n_samples(self):
        return self._data.shape[0]

    @property
    def label_cardinality(self):
        if self.is_multilabel:
            return np.mean(np.sum(self._target, axis=1))
        return 1

    @property
    def label_density(self):
        return self.label_cardinality / self.n_classes

    @property
    def label_diversity(self):
        return len(np.unique(self._target, axis=0))

    @property
    def all_attributes(self):
        return {"Name": self.name,
                "Is semisupervised": self.is_semisupervised,
                "Is multilabel": self.is_multilabel,
                "Datashape": self.data.shape,
                "Feature names": self.feature_names,
                "Target shape": self._target.shape,
                "Target classes": self.classes,
                "Target labels": self.names,
                "Target counts": self.counts,
                "Label cardinality": self.label_cardinality,
                "Label density": self.label_density,
                "Label diversity": self.label_diversity}

    def __str__(self):
        return "\n".join('{} = {}'.format(key, value) for key, value in
                         self.all_attributes.items())


from .synthetic.fruits import get_fruits
from .synthetic.sklearnmultilabel import get_sklearn_multilabel
from .synthetic.mlgaussians import get_mlgaussians

datasets_synthetic = {'fruits': get_fruits,
                      'sklearn-multilabel': get_sklearn_multilabel,
                      'mlgaussians': get_mlgaussians}

class Data(object):
    uci_nan = -2147483648
    # TODO mldata is not working anymore, I need to change each of this calls
    # to download it from a copy in my repo with loadmat('sonar.mat')
    pydataset_names = ['iris', 'aids', 'turnout']

    pydataset_not_working = ['diabetes']

    openml_names = {
                    'ecoli':'ecoli',
                    'birds': 'birds',
                    'enron': 'enron',
                    'emotions': 'emotions',
                    'scene': 'scene',
                    'yeast': 'yeast',
                    'reuters': 'reuters',
                    'slashdot': 'slashdot',
                    'image': 'image',
                    'genbase': 'genbase',
                    'langlog': 'langlog'
                    }

    openml_not_working = {
                    #'spam':'uci-20070111 spambase', # not working
                    'mushroom':'uci-20070111 mushroom',
                    # To be added:
                    'breast-cancer-w':'uci-20070111 wisconsin',
                    # Need preprocessing :
                    'auslan':'',
                    # Needs to be generated
                    'led7digit':'',
                    'yeast':'',
                    # Needs permission from ml-repository@ics.uci.edu
                    'lymphography':'',
                    # HTTP Error 500 in mldata.org
                    'satimage':'satimage',
                    'nursery':'uci-20070111 nursery',
                    'hypothyroid':'uci-20070111 hypothyroid',
                    'glass':'glass',
                    'heart-statlog':'datasets-UCI heart-statlog',
                    'ionosphere':'ionosphere',
                    'letter':'letter',
                    'mfeat-karhunen':'uci-20070111 mfeat-karhunen',
                    'mfeat-morphological':'uci-20070111 mfeat-morphological',
                    'mfeat-zernike':'uci-20070111 mfeat-zernike',
                    'optdigits':'uci-20070111 optdigits',
                    'pendigits':'uci-20070111 pendigits',
                    'sonar':'sonar',
                    'vehicle':'vehicle',
                    'waveform-5000':'datasets-UCI waveform-5000',
                    'scene-classification':'scene-classification',
                    'tic-tac':'uci-20070111 tic-tac-toe',
                    'MNIST':'MNIST (original)',
                    'autos':'uci-20070111 autos',
                    'car':'uci-20070111 car',
                    'cleveland':'uci-20070111 cleveland',
                    'dermatology':'uci-20070111 dermatology',
                    'flare':'uci-20070111 solar-flare_2',
                    'page-blocks':'uci-20070111 page-blocks',
                    'segment':'datasets-UCI segment',
                    'shuttle':'shuttle',
                    'vowel':'uci-20070111 vowel',
                    'zoo':'uci-20070111 zoo',
                    'abalone':'uci-20070111 abalone',
                    'balance-scale': 'uci-20070111 balance-scale',
                    'credit-approval':'uci-20070111 credit-a',
                    'german':'German IDA',
                    'hepatitis':'uci-20070111 hepatitis',
                    'lung-cancer':'Lung Cancer (Michigan)'
            }

    def __init__(self, dataset_names=None, data_home='./datasets/',
                 load_all=False, shuffle=True, random_state=None):
        self.data_home = data_home
        self.datasets = {}

        if load_all:
            dataset_names = Data.openml_names.keys()
            self.load_datasets_by_name(dataset_names)
        elif dataset_names is not None:
            self.load_datasets_by_name(dataset_names)

        if shuffle:
            for name in self.datasets.keys():
                self.datasets[name].shuffle(random_state=random_state)


    def load_datasets_by_name(self, names):
        for name in names:
            dataset = self.get_dataset_by_name(name)
            if dataset is not None:
                self.datasets[name] = self.get_dataset_by_name(name)
            else:
                warnings.simplefilter('always', UserWarning)
                warnings.warn(("Dataset '{}' not currently available.".format(name)),
                              UserWarning)

    def download_file_content(self, url):
        response = urllib2.urlopen(url, timeout = 5)
        return response.read()

    def save_file_content(self, filename, content):
        with open( filename, 'w' ) as f:
            f.write( content )

    def check_file_and_download(self, file_path, url):
        if not isfile(file_path):
            content = self.download_file_content(url)
            self.save_file_content(file_path, content)

    def get_dataset_by_name(self, name):
        if name in Data.openml_names.keys():
            return self.get_openml_dataset(name)
        elif name in Data.pydataset_names:
            return self.get_pydataset_dataset(name)
        elif name in datasets_synthetic.keys():
            return datasets_synthetic[name]()
        elif name == 'spambase':
            file_path = self.data_home+'spambase.data'
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
            self.check_file_and_download(file_path, url)

            data = np.genfromtxt(file_path, delimiter=',')
            target = data[:,-1]
            data = data[:,0:-1]
        elif name == 'horse':
            file_path = self.data_home+'horse-colic.data'
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.data"
            self.check_file_and_download(file_path, url)

            data = np.genfromtxt(file_path)
            target = data[:,23]
            data = np.delete(data, 23, axis=1)
            data = self.substitute_missing_values(data, column_mean=True)
        elif name == 'libras-movement':
            file_path = self.data_home+'movement_libras.data'
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data"
            self.check_file_and_download(file_path, url)

            data = np.genfromtxt(file_path, delimiter=',')
            target = data[:,-1]
            data = np.delete(data, -1, axis=1)
        elif name == 'mushroom':
            file_path = self.data_home+'agaricus-lepiota.data'
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
            self.check_file_and_download(file_path, url)

            data = np.genfromtxt(file_path, delimiter=',', dtype=np.object_)
            target = data[:,0]
            data = np.delete(data, 0, axis=1)
            for i in range(data.shape[1]):
                data[:,i] = self.nominal_to_float(data[:,i])
            data = data.astype(float)
            data = self.substitute_missing_values(data, column_mean=True)
        elif name == 'landsat-satellite':
            file_path = self.data_home+'sat.trn'
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn"
            self.check_file_and_download(file_path, url)

            file_path = self.data_home+'sat.tst'
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst"
            self.check_file_and_download(file_path, url)

            data_train = np.genfromtxt(self.data_home+'sat.trn')
            data_test = np.genfromtxt(self.data_home+'sat.tst')

            target = np.hstack((data_train[:,-1], data_test[:,-1]))
            data = np.vstack((np.delete(data_train, -1, axis=1),
                              np.delete(data_test, -1, axis=1)))
        elif name == 'yeast':
            file_path = self.data_home+'yeast.data'
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
            self.check_file_and_download(file_path, url)

            target = np.genfromtxt(file_path, usecols=9, dtype=str)
            data = np.genfromtxt(self.data_home+'yeast.data')[:,1:-1]
        elif name == 'wdbc':
            file_path = self.data_home+'wdbc.data'
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
            self.check_file_and_download(file_path, url)

            data = np.genfromtxt(file_path, delimiter=',')
            data = np.delete(data, (0,1), axis=1)
            target = np.genfromtxt(file_path, delimiter=',', usecols=1,
                                   dtype=str)
        elif name == 'wpbc':
            file_path = self.data_home+'wpbc.data'
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data"
            self.check_file_and_download(file_path, url)

            data = np.genfromtxt(file_path, delimiter=',')
            data = np.delete(data, (0,1), axis=1)
            target = np.genfromtxt(file_path, delimiter=',', usecols=1,
                                   dtype=str)
            data, target = self.remove_rows_with_missing_values(data, target)
        else:
            return None
        return Dataset(name, data, target)

    def get_pydataset_dataset(self, name):
        try:
            data = pydataset_data(name)
        except Exception as e:
            print(e)
            return None

        feature_names = None
        if name == 'iris':
            target = data['Species'].values
            feature_names = ['Sepal.Length', 'Sepal.Width', 'Petal.Length',
                             'Petal.Width']
            data = data[feature_names].values
        elif name == 'aids':
            target = data['adult'].values
            feature_names = ['infect', 'induct']
            data = data[feature_names].values
        elif name == 'turnout':
            target = data['vote'].values
            feature_names = ['race', 'age', 'educate', 'income']
            data['race'] = data['race'].astype('category')
            data['race'] = data['race'].cat.codes
            data = data[feature_names].values
        else:
            ValueError('Dataset ' + name + 'not implemented yet.')

        return Dataset(name, data, target, feature_names)

    def get_openml_dataset(self, name):
        try:
            openml = fetch_openml(Data.openml_names[name],
                                  data_home=self.data_home)
            #dataset = openml.datasets.get_dataset(Data.openml_names[name],
            #                                     data_home=self.data_home)
            #data, target, _, _ = dataset.get_data(dataset_format="dataframe")
            #return Dataset(name, data, target)
        except Exception as e:
            print(e)
            return None

        # TODO adapt code using this method
        #MAPPING = {'ecoli': lambda x: x.target.T, x.data,
        #           'diabetes': lambda x: x.data, x.target,
        #           'optdigits': lambda x: x.data[:,:-1], x.data[:,-1],
        #           'pendigits': lambda x: x.data[:,:-1], x.data[:,-1],
        #           'waveform-5000': lambda x: x.target.T, x.data,
        #           'heart-statlog': lambda x: np.hstack([x.target.T, x.data, x['int2'].T]), x['class'],
        #           'mfeat-karhunen': lambda x: x.target.T, x.data,
        #           'mfeat-zernike': lambda x: x.target.T, x.data,
        #           'mfeat-morphological': lambda x: x.target.T, x.data,
        #           'waveform-5000': lambda x: x.target.T, x.data}
        #data, target = MAPPING[name](openml)

        if name=='ecoli':
            data = openml.data
            target = openml.target
        elif name=='birds':
            data = openml.data
            target = openml.target == 'TRUE'
        elif name=='langlog':
            data = openml.data
            target = openml.target == 'TRUE'
        elif name=='enron':
            data = openml.data
            target = openml.target == 'TRUE'
        elif name=='emotions':
            data = openml.data
            target = openml.target == 'TRUE'
        elif name=='diabetes':
            data = openml.data
            target = openml.target
        elif name=='optdigits':
            data = openml.data[:,:-1]
            target = openml.data[:,-1]
        elif name=='pendigits':
            data = openml.data[:,:-1]
            target = openml.data[:,-1]
        elif name=='waveform-5000':
            data = openml.target.T
            target = openml.data
        elif name=='heart-statlog':
            data = np.hstack([openml['target'].T, openml.data, openml['int2'].T])
            target = openml['class']
        elif name=='mfeat-karhunen':
            data = openml.target.T
            target = openml.data
        elif name=='mfeat-zernike':
            data = openml.target.T
            target = openml.data
        elif name=='mfeat-morphological':
            data = openml.target.T
            target = openml.data
        elif name=='scene-classification':
            data = openml.data
            target = openml.target.toarray()
            target = target.transpose()[:,4]
        elif name=='genbase':
            data = openml.data
            target = openml.target.values
        elif name=='tic-tac':
            n = np.alen(openml.data)
            data = np.hstack((openml.data.reshape(n,1),
                                     np.vstack([openml[feature] for feature in
                                                openml.keys() if 'square' in
                                                feature]).T,
                                     openml.target.reshape(n,1)))
            for i, value in enumerate(np.unique(data)):
                data[data==value] = i
            data = data.astype(float)
            target = openml.Class.reshape(n,1)
        elif name=='autos':
            target = openml.int5[5,:].reshape(-1,1)
            data = np.hstack((
                      openml['target'].reshape(-1,1),
                      self.nominal_to_float(openml['data'].reshape(-1,1)),
                      self.nominal_to_float(openml['fuel-type'].reshape(-1,1)),
                      self.nominal_to_float(openml['aspiration'].reshape(-1,1)),
                      self.nominal_to_float(openml['num-of-doors'].reshape(-1,1)),
                      self.nominal_to_float(openml['body-style'].reshape(-1,1)),
                      self.nominal_to_float(openml['drive-wheels'].reshape(-1,1)),
                      self.nominal_to_float(openml['engine-location'].reshape(-1,1)),
                      openml['double1'].T.reshape(-1,4),
                      openml['int2'].reshape(-1,1),
                      self.nominal_to_float(openml['engine-type'].reshape(-1,1)),
                      self.nominal_to_float(openml['num-of-cylinders'].reshape(-1,1)),
                      openml['int3'].reshape(-1,1),
                      self.nominal_to_float(openml['fuel-system'].reshape(-1,1)),
                      openml['double4'].T.reshape(-1,3),
                      openml['int5'][:-1,:].T.reshape(-1,5)
                              ))
            data, target = self.remove_rows_with_missing_values(data, target)
        elif name=='car':
            target = openml['class']
            feature_names = ['data', 'target', 'doors', 'persons', 'lug_boot',
                             'safety']
            data = np.hstack([
                    self.nominal_to_float(openml[f_name].reshape(-1,1))
                        for f_name in feature_names])
        elif name=='cleveland':
            target = openml.int2.reshape(-1,1)
            data = np.hstack((openml.target.T, openml.data))
            data, target = self.remove_rows_with_missing_values(data, target)
        elif name=='dermatology':
            target = openml.data[:,-1]
            data = openml.data[:,:-1]
            data, target = self.remove_rows_with_missing_values(data, target)
        elif name=='flare':
            target = openml.target
            data = openml['int0'].T

            # TODO this dataset is divided in two files, see more elegant way
            # to add it
            try:
                openml = fetch_openml('uci-20070111 solar-flare_1')
            except Exception:
                return None

            target = np.hstack((target, openml.target))
            data = np.vstack((data, openml['int0'].T))
        elif name=='nursery':
            raise Exception('Not currently available')
        elif name=='page-blocks':
            data = np.hstack((openml['target'].T, openml['data'],
                              openml['int2'].T))
            target = data[:,-1]
            data = data[:,:-1]
        elif name=='satimage':
            raise Exception('Not currently available')
        elif name=='segment':
            target = openml['class'].reshape(-1,1)
            data = np.hstack((openml['int2'].T, openml['data'],
                              openml['target'].T, openml['double3'].T))
        elif name=='vowel':
            target = openml['Class'].T
            # We are not using the extra features implicit in the dataset:
            # target: {training, test}
            # data: Name of the participant
            # sex: sex of the participant
            data = openml['double0'].T
        elif name=='zoo':
            target = openml['type'].reshape(-1,1)
            feature_names = ['aquatic', 'domestic', 'eggs', 'backbone',
                             'feathers', 'data', 'milk', 'tail',
                             'airborne', 'toothed', 'catsize', 'venomous',
                             'fins', 'predator', 'breathes']
            data = np.hstack([
                    self.nominal_to_float(openml[f_name].reshape(-1,1))
                        for f_name in feature_names])
            data = np.hstack((data, openml['int0'].T))
        elif name=='abalone':
            target = openml.target
            data = np.hstack((openml['data'], openml['int1'].T))
        elif name=='balance-scale':
            target = openml.data
            data = openml.target.T
        elif name=='credit-approval':
            target = openml['class'].T
            data = self.openml_to_numeric_matrix(openml, 690,
                                                 exclude=['class'])
            data, target = self.remove_rows_with_missing_values(data, target)
        elif name=='hepatitis':
            target = openml['Class'].T
            data = self.openml_to_numeric_matrix(openml, 155,
                                                 exclude=['Class'])
            data = self.substitute_missing_values(data, column_mean=True)
        elif name=='lung-cancer':
            target = openml['Class'].T
            data = self.openml_to_numeric_matrix(openml, 96,
                                                 exclude=['Class'])
        elif name=='reuters':
            target = openml['target'] == 'TRUE'
            data = openml['data']
        elif name=='image':
            target = openml['target'] == 'TRUE'
            data = openml['data']
        elif name=='genbase':
            # FIXME returning a ValueError
            target = openml['target'] == 'TRUE'
            data = openml['data']
        elif name=='slashdot':
            target = openml['target'] == 'TRUE'
            data = openml['data']
        else:
            try:
                data = openml.data
                target = openml.target
            except Exception as e:
                print(e)
                return None

        return Dataset(name, data, target)

    def openml_to_numeric_matrix(self, openml, n_samples, exclude=[]):
        """converts an openml object into a matrix

        for each value in the openml dictionary it is reshaped to contain the
        first dimension as a number of samples and the second as number of
        features. If the value contains numerical data it is not preprocessed.
        If the value contains any other type np.object_ it is transformed to
        numerical and all the missing values marked with '?' or 'nan' are
        substituted by np.nan.
        Args:
            openml (dictionary with some numpy.array): feature strings.

        Returns:
            (array-like, shape = [n_samples, n_features]): floats.
        """
        first_column = True
        for key, submatrix in openml.items():
            if key not in exclude and type(submatrix) == np.ndarray:
                new_submatrix = np.copy(submatrix)

                if new_submatrix.shape[0] != n_samples:
                    new_submatrix = new_submatrix.T

                if new_submatrix.dtype.type == np.object_:
                    new_submatrix = self.nominal_to_float(new_submatrix)

                if first_column:
                    matrix = new_submatrix.reshape(n_samples, -1)
                    first_column = False
                else:
                    matrix = np.hstack((matrix,
                                        new_submatrix.reshape(n_samples, -1)))
        return matrix


    def nominal_to_float(self, x, missing_values=['nan', '?']):
        """converts an array of nominal features into floats

        Missing values are marked with the string 'nan' and are converted to
        numpy.nan

        Args:
            x (array-like, shape = [n_samples, 1]): feature strings.

        Returns:
            (array-like, shape = [n_samples, 1]): floats.
        """
        new_x = np.empty_like(x, dtype=float)
        x = np.squeeze(x)
        names = np.unique(x)
        substract = 0
        for i, name in enumerate(names):
            if name in missing_values:
                new_x[x==name] = np.nan
                substract += 1
            else:
                new_x[x==name] = i - substract
        return new_x

    def number_of_missing_values(self,data):
        return np.logical_or(np.isnan(data), data == self.uci_nan).sum()

    def row_indices_with_missing_values(self,data):
        return np.logical_or(np.isnan(data),
                             data == self.uci_nan).any(axis=1)

    def remove_rows_with_missing_values(self, data, target):
        missing = self.row_indices_with_missing_values(data)
        data = data[~missing]
        target = target[~missing]
        return data, target

    def remove_columns_with_missing_values(self, data, n_columns=1):
        for i in range(n_columns):
            index = np.isnan(data).sum(axis=0).argmax()
            data = np.delete(data, index, axis=1)
        return data


    def substitute_missing_values(self, data, fix_value=0, column_mean=False):
        for i in range(data.shape[1]):
            index = np.where(np.isnan(data[:,i]))
            if column_mean:
                mean = np.nanmean(data[:,i])
                data[index,i] = mean
            else:
                data[index,i] = fix_value
        return data


    def sumarize_datasets(self, name=None):
        if name is not None:
            dataset = self.datasets[name]
            dataset.print_summary()
        else:
            for name, dataset in self.datasets.items():
                dataset.print_summary()

def test_datasets(dataset_names):
    from sklearn.svm import SVC
    data = Data(dataset_names=dataset_names)

    def separate_sets(x, y, test_fold_id, test_folds):
        x_test = x[test_folds == test_fold_id, :]
        y_test = y[test_folds == test_fold_id]

        x_train = x[test_folds != test_fold_id, :]
        y_train = y[test_folds != test_fold_id]
        return [x_train, y_train, x_test, y_test]

    n_folds = 2
    accuracies = {}
    for name, dataset in data.datasets.items():
        dataset.print_summary()
        skf = StratifiedKFold(dataset.target, n_folds=n_folds, shuffle=True)
        test_folds = skf.test_folds
        accuracies[name] = np.zeros(n_folds)
        test_fold = 0
        for train_idx, test_idx in skf.split(X=dataset.data, y=dataset.target):
            x_train, y_train = dataset.data[train_idx], dataset.target[train_idx]
            x_test, y_test = dataset.data[test_idx], dataset.target[test_idx]

            svc = SVC(C=1.0, kernel='rbf', degree=1, tol=0.01)
            svc.fit(x_train, y_train)
            prediction = svc.predict(x_test)
            accuracies[name][test_fold] = 100*np.mean((prediction == y_test))
            print("Acc = {0:.2f}%".format(accuracies[name][test_fold]))
            test_fold += 1
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


#class openml(Data):
#    def __init__(self, data_home='./datasets/', load_all=False):
#        warnings.simplefilter('always', DeprecationWarning)
#        warnings.warn(('This Class is going to be deprecated in a future '
#                       'version, please use cwc.data_wrappers.Data instead.'),
#                      DeprecationWarning)
#        self.data_home = data_home
#        self.datasets = {}
#
#        if load_all:
#            for key in openml.openml_names.keys():
#                self.datasets[key] = self.get_dataset(key)

if __name__=='__main__':
    test()
