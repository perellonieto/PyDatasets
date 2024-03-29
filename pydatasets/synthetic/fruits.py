import numpy as np
from scipy.stats import multivariate_normal
from sklearn.utils import shuffle

from pydatasets.datasets import Dataset
from pydatasets.random import temp_seed

class fruit_and_insect_generator(object):
    def __init__(self, fruit_mean, fruit_cov, insect_mean, insect_cov, mu0,
                 mu1, random_state=None):
        self.fruit_gen = multivariate_normal(mean=fruit_mean, cov=fruit_cov)
        self.insect_gen = multivariate_normal(mean=insect_mean, cov=insect_cov)
        self.mu0 = mu0
        self.mu1 = mu1
        self.random_state = random_state
        self.repetition = 0

    def rvs(self, n_samples):
        x = self.fruit_gen.rvs(n_samples)
        x = np.clip(x, 0.1, None)
        q_x = self.insect_gen.pdf(x) / self.insect_gen.pdf(self.insect_gen.mean)
        p_insect = (1 - q_x)*self.mu0 + q_x*self.mu1
        if self.random_state is None:
            self.random_state = np.random.get_state()[1][0]
        with temp_seed(self.random_state + self.repetition):
            insects = np.random.binomial(1, p_insect).astype(bool)
            self.repetition += 1
        return x, insects

fruits_kwargs = {'cucumbers': dict(fruit_mean=[4, 7],
                                   fruit_cov=[[1, 2], [1, 3]],
                                   insect_mean=[4, 9],
                                   insect_cov=[[1, 0], [0, 1]],
                                   mu0=0, mu1=0.8),
                 'oranges': dict(fruit_mean=[4, 4],
                                 fruit_cov=[[0.6, 0.5], [0, 0.6]],
                                 insect_mean=[4, 5],
                                 insect_cov=[[0.4, 0], [0, 0.4]],
                                 mu0=0, mu1=0.9)
                }

def get_fruits(samples_per_class=10000, random_state=None):
    x1, y1 = fruit_and_insect_generator(
        random_state=random_state,
        **fruits_kwargs['oranges']).rvs(samples_per_class)
    x2, y2 = fruit_and_insect_generator(
        random_state=random_state,
        **fruits_kwargs['cucumbers']).rvs(samples_per_class)

    Y = np.concatenate([y1, y2])
    Y = np.vstack([Y,
                   np.concatenate([np.ones_like(y1), np.zeros_like(y2)])]).T
    X = np.concatenate([x1, x2])
    dataset = Dataset('fruits', X, Y, feature_names=['width', 'height'],
                      shuffle=True, random_state=random_state)
    return dataset
