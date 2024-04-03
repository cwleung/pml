from distributions.base import Distribution
import numpy as np


class Poisson(Distribution):
    def __init__(self, rate):
        self.rate = rate

    def __add__(self, other):
        return Poisson(rate=self.rate + other.rate)

    def __matmul__(self, other):
        if other.shape[1] != 1:
            raise ValueError(f"Cannot multiply with matrix of shape {other.shape}")
        return Poisson(rate=other @ self.rate)

    def sample(self, n_samples=1):
        return np.random.poisson(lam=self.rate, size=n_samples)

    def log_pdf(self, x):
        return x * np.log(self.rate) - self.rate - np.log(np.math.factorial(x))

    def kl_divergence(self, other):
        return self.rate - other.rate - other.rate * (np.log(self.rate) - np.log(other.rate))

    def plot(self, n_samples=1000):
        """Plot the distribution"""
        import matplotlib.pyplot as plt
        plt.hist(self.sample(n_samples), bins=20)
        plt.show()