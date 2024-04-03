import numpy as np

from distributions.base import Distribution


class Bernoulli(Distribution):
    def __init__(self, p):
        self.p = p

    def __add__(self, other):
        return Bernoulli(p=self.p + other.p)

    def __matmul__(self, other):
        if other.shape[1] != 1:
            raise ValueError(f"Cannot multiply with matrix of shape {other.shape}")
        return Bernoulli(p=other @ self.p)

    def sample(self, n_samples=1):
        return np.random.binomial(n=1, p=self.p, size=n_samples)

    def log_pdf(self, x):
        return np.log(np.where(x == 1, self.p, 1 - self.p))

    def kl_divergence(self, other):
        return self.p * np.log(self.p / other.p) + (1 - self.p) * np.log((1 - self.p) / (1 - other.p))

    def plot(self, n_samples=1000):
        """Plot the distribution"""
        import matplotlib.pyplot as plt
        plt.hist(self.sample(n_samples), bins=2)
        plt.show()
