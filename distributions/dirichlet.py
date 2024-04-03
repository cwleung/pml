from distributions.base import Distribution

import numpy as np


class Dirichlet(Distribution):
    def __init__(self, alpha):
        self.alpha = alpha

    def __add__(self, other):
        return Dirichlet(alpha=self.alpha + other.alpha)

    def __matmul__(self, other):
        if other.shape[1] != 1:
            raise ValueError(f"Cannot multiply with matrix of shape {other.shape}")
        return Dirichlet(alpha=other @ self.alpha)

    def sample(self, n_samples=1):
        return np.random.dirichlet(alpha=self.alpha, size=n_samples)

    def log_pdf(self, x):
        return (
            np.sum(
                (self.alpha - 1) * np.log(x)
                - np.log(np.math.gamma(np.sum(self.alpha), axis=1))
                + np.sum(np.log(np.math.gamma(self.alpha)), axis=1),
                axis=1,
            )
        )

    def pdf(self, x):
        return np.exp(self.log_pdf(x))

    def plot(self):
        import matplotlib.pyplot as plt

        print(f"Distribution: {self.alpha}")
        plt.bar(range(len(self.alpha)), self.alpha)
        plt.show()

    def kl_divergence(self, other):
        return np.sum((self.alpha - other.alpha) * (np.log(self.alpha) - np.log(other.alpha)))