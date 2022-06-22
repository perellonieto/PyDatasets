from pydatasets.datasets import Data

name_list = ['emotions', 'enron', 'iris', 'birds']
data = Data(dataset_names=name_list)
for name in name_list:
    print(data.datasets[name])
