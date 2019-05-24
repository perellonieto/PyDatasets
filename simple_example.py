from pydatasets.datasets import Data

name = 'turnout'
data = Data(dataset_names=[name])
print(data.datasets[name])
