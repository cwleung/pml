import numpy as np

from distributions.base import Distribution


class Categorical(Distribution):

    def __init__(self, p):
        self.p = p

    def __add__(self, other):
        return Categorical(p=self.p + other.p)

    def __matmul__(self, other):
        if other.shape[1] != 1:
            raise ValueError(f"Cannot multiply with matrix of shape {other.shape}")
        return Categorical(p=other @ self.p)

    def sample(self, n_samples=1):
        return np.random.choice(self.p, size=n_samples, p=self.p)

    def rsmaple(self, n_samples=1):
        return np.random.choice(self.p, size=n_samples, p=self.p)

    def kl_divergence(self, other):
        return np.sum(self.p * np.log(self.p / other.p))

    def log_pdf(self, x: np.array):
        x = x.astype(int)
        return np.log(self.p[x])

    @staticmethod
    def mle(data: np.ndarray, n_categories=None, alpha=1.0):
        """
        Maximum likelihood estimation of the Categorical distribution with Laplace smoothing.

        :param n_categories: Number of classes
        :param data: Data samples (each row is a sample). (n_samples, n_features)
        :param alpha: Laplace smoothing parameter (default is 1.0)
        :return: Categorical object with updated probabilities
        """
        if data.ndim == 1:
            likelihood = np.zeros(int(n_categories))
            for k in range(int(n_categories)):
                likelihood[k] = ((data == k).sum() + alpha) / (len(data) + alpha * n_categories)
            return Categorical(p=likelihood)
        else:
            raise ValueError("Input data should be 1D array.")

    def __repr__(self):
        return f"Categorical(p={self.p})"
