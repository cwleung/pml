from distributions.base import Distribution
import numpy as np


class Binomial(Distribution):
    def __init__(self, n, p):
        self.n = n
        self.p = p

    def __add__(self, other):
        return Binomial(n=self.n + other.n, p=self.p + other.p)

    def __matmul__(self, other):
        if other.shape[1] != 1:
            raise ValueError(f"Cannot multiply with matrix of shape {other.shape}")
        return Binomial(n=other @ self.n, p=other @ self.p)

    def sample(self, n_samples=1):
        return np.random.binomial(n=self.n, p=self.p, size=n_samples)

    def log_pdf(self, x):
        return np.sum(np.log(np.where(x == 1, self.p, 1 - self.p)), axis=1)

    def kl_divergence(self, other):
        return self.p * np.log(self.p / other.p) + (1 - self.p) * np.log((1 - self.p) / (1 - other.p))

    def plot(self, n_samples=1000):
        """Plot the binomial distribution"""
        import matplotlib.pyplot as plt
        plt.hist(self.sample(n_samples), bins=2)
        plt.show()