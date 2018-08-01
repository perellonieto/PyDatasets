from pydatasets import Data

dataset_name = 'iris'
data = Data(data_home='./datasets/')
dataset = data.get(dataset_name)
