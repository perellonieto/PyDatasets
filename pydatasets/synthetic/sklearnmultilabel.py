from pydatasets.datasets import Dataset
from sklearn.datasets import make_multilabel_classification


def get_sklearn_multilabel(samples_per_class=10000, n_features=30, n_classes=5,
                          n_labels=None, length=50, random_state=None):
    if n_labels is None:
        n_labels = int(n_classes/2)
    n_samples = int(samples_per_class*n_classes)
    X, Y = make_multilabel_classification(n_samples=n_samples,
                                          n_features=n_features,
                                          n_classes=n_classes,
                                          n_labels=n_labels, length=length,
                                          random_state=random_state)

    dataset = Dataset('sklearn-multilabel', X, Y, shuffle=True,
                      random_state=random_state)
    return dataset
