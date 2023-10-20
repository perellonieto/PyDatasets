from pydatasets.datasets import Dataset

import numpy as np
import scipy.stats as stats


def load_blob(n_sample, n_label):
    print("Deprecated. Import with from multicalpy.datasets import load_blob")
    L = 2 ** n_label
    pi_power = stats.dirichlet.rvs(np.ones(L)).reshape((2,) * n_label)
    pi = pi_power.reshape((2,) * n_label)
    Y_power = stats.multinomial.rvs(p=pi_power.ravel(), n=1, size=n_sample)
    tmp_Y = Y_power.reshape((n_sample,) + (2,) * n_label)
    Y = np.hstack([np.sum(np.take(tmp_Y, 1, j+1), axis=tuple(np.arange(n_label)[1:])).reshape(-1, 1) for j in range(n_label)])

    mu = np.random.randn(2**n_label, 2) * 2.0
    X = np.zeros((n_sample, 2))
    for i in range(2**n_label):
        idx = (Y_power[:, i]==1)
        X[idx, :] = np.random.randn(np.sum(idx), 2) + mu[i, :]

    return X, Y, Y_power, pi


def get_dirichlet_blobs(n_samples=10000, random_state=None):
    X, Y, Y_power, pi = load_blob(n_samples, 3)
    dataset = Dataset('dirichlet_blobs', X, Y, feature_names=['X1', 'X2'],
                      shuffle=True, random_state=random_state)
    return dataset

