__docformat__ = 'restructedtext en'
import numpy as np

from .dataset import Dataset
from .data import Data

__author__ = "Miquel Perello Nieto"
__credits__ = ["Miquel Perello Nieto"]

__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Miquel Perello Nieto"
__email__ = "miquel@perellonieto.com"
__status__ = "Development"

import urllib2
from os.path import isfile

def test():
    dataset_name = 'iris'
    data = Data(data_home='./datasets/')
    dataset = data.get(dataset_name)


if __name__=='__main__':
    test()
