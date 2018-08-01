__docformat__ = 'restructedtext en'
import warnings
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import StratifiedKFold
import numpy as np

from dataset import Dataset

__author__ = "Miquel Perello Nieto"
__credits__ = ["Miquel Perello Nieto"]

__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Miquel Perello Nieto"
__email__ = "miquel@perellonieto.com"
__status__ = "Development"

import urllib2
from os.path import isfile

from .instances.iris import Iris

class Data(object):
    def __init__(self, data_home='./datasets/'):
        self.data_home = data_home

    def get(self, name):
        if name=='iris':
            return Iris()
