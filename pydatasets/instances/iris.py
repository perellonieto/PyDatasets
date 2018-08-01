__docformat__ = 'restructedtext en'
from ..dataset import MldataDataset

__author__ = "Miquel Perello Nieto"
__credits__ = ["Miquel Perello Nieto"]
__email__ = "miquel@perellonieto.com"

class Iris(MldataDataset):
    def __init__(self):
        self.name = 'iris'
        self.mldata_name = 'iris'
        MldataDataset.__init__(self)
