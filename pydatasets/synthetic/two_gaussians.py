import numpy as np
from scipy.stats import multivariate_normal
from sklearn.utils import shuffle

from pydatasets.datasets import Dataset


class MultivariateGaussian(object):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def rvs(self, size):
        return np.array(self.dist.rvs(size=size))

    @property
    def dist(self):
        return multivariate_normal(mean=self.mean, cov=self.cov)


def get_two_gaussians(samples_per_class=10000, random_state=None):
    N = samples_per_class
    mg_list = [MultivariateGaussian(mean=[1, 1],
                                    cov=[[1, 0.8], [0.8, 1]]),
               MultivariateGaussian(mean=[-1, -1],
                                    cov=[[1, 0.8], [0.8, 1]])
              ]

    X = np.concatenate([mg.rvs(size=N) for mg in mg_list])

    Y = np.vstack((np.hstack((np.ones(N), np.zeros(N))),
                  (np.hstack((np.zeros(N), np.ones(N)))))).T


    dataset = Dataset('twogaussians', X, Y, feature_names=['X1', 'X2'],
                      shuffle=True, random_state=random_state)
    return dataset
