from pydatasets.datasets import Data

name = 'ecoli'
data = Data(dataset_names=[name])
print(data.datasets[name])
