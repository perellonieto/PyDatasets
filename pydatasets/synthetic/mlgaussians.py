import numpy as np
from scipy.stats import multivariate_normal
from sklearn.utils import shuffle

from pydatasets.datasets import Dataset

class dependent_gaussians_generator(object):
    def __init__(self, g1_mean, g1_cov, g2_mean, g2_cov, mu0, mu1):
        self.g1_gen = multivariate_normal(mean=g1_mean, cov=g1_cov)
        self.g2_gen = multivariate_normal(mean=g2_mean, cov=g2_cov)
        self.mu0 = mu0
        self.mu1 = mu1

    def rvs(self, n_samples):
        x = self.g1_gen.rvs(n_samples)
        q_x = self.g2_gen.pdf(x) / self.g2_gen.pdf(self.g2_gen.mean)
        p_g2 = (1 - q_x)*self.mu0 + q_x*self.mu1
        g2s = np.random.binomial(1, p_g2).astype(bool)
        return x, g2s

g1s = {'C1': dependent_gaussians_generator(g1_mean=[-3, 0],
                                       g1_cov=[[1, 0], [0, 1]],
                                       g2_mean=[-2, -2],
                                       g2_cov=[[.2, 0], [0, .2]],
                                       mu0=0.3, mu1=0.99),
       'C2': dependent_gaussians_generator(g1_mean=[2, -2],
                                     g1_cov=[[2, 0], [0, 2]],
                                     g2_mean=[2, 0],
                                     g2_cov=[[0.5, 0], [0, 0.5]],
                                     mu0=0, mu1=0.95),
       'C3': dependent_gaussians_generator(g1_mean=[2, 2],
                                     g1_cov=[[2, 0], [0, 2]],
                                     g2_mean=[2, 0],
                                     g2_cov=[[0.5, 0], [0, 0.5]],
                                     mu0=0.4, mu1=0.8)
         }

def get_mlgaussians(samples_per_class=10000):
    x1, y1 = g1s['C1'].rvs(samples_per_class)
    x2, y2 = g1s['C2'].rvs(samples_per_class)
    x3, y3 = g1s['C3'].rvs(samples_per_class)

    Y = np.vstack([np.concatenate([np.ones_like(y1),    # Label class 1
                                   np.zeros_like(y2),
                                   np.zeros_like(y3)]),
                   np.concatenate([np.zeros_like(y1),
                                   np.ones_like(y2),    # Label class 2
                                   np.zeros_like(y3)]),
                   np.concatenate([np.zeros_like(y1),
                                   np.zeros_like(y2),
                                   np.ones_like(y3)]),  # Label class 3
                   np.concatenate([y1,                  # Label subclass 1
                                    np.zeros_like(y2),
                                    np.zeros_like(y3)]),
                   np.concatenate([np.zeros_like(y1),
                                    y2,                 # Label subclass 2 & 3
                                    y3])
                   # np.concatenate([np.zeros_like(y1),
                   #                  y2,                 # Label subclass 2
                   #                  np.zeros_like(y3)]),
                   # np.concatenate([np.zeros_like(y1),
                   #                  np.zeros_like(y2),
                   #                  y3])                # Label subclass 3
                  ]).T
    X = np.concatenate([x1, x2, x3])
    dataset = Dataset('mlgaussians', X, Y, feature_names=['X1', 'X2'],
                      shuffle=True)
    return dataset
