import openml

def load_yeast(shuffle=False, random_state=None):
    from pydatasets.datasets import Dataset
    dataset = openml.datasets.get_dataset(40597)
    df, _, _, attributes = dataset.get_data(dataset_format='dataframe')
    attributes = [col for col in df if col.startswith('Att')]
    classes = [col for col in df if col.startswith('Class')]
    X = df[attributes].values
    Y = df[classes].values.astype(int)
    dataset = Dataset('yeast-ml', X, Y, feature_names=attributes,
                      shuffle=shuffle, random_state=random_state)
    return dataset
