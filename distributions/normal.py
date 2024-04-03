import numpy as np

from distributions.base import Distribution


class Gaussian(Distribution):
    """Univariate Gassian distribution"""

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self, n_samples=1):
        return np.random.normal(self.mu, self.sigma, size=n_samples)

    def log_pdf(self, x):
        """
        -0.5 * log(2 * pi * sigma^2) - 0.5 * ((x - mu)^2) / sigma^2
        """
        return -0.5 * np.log(2 * np.pi * self.sigma ** 2) - 0.5 * ((x - self.mu) ** 2) / (self.sigma ** 2)

    def kl_divergence(self, other):
        """
        0.5 * (log(sigma2 / sigma1) + (sigma1^2 + (mu1 - mu2)^2) / sigma2^2 - 1)
        """
        return 0.5 * (np.log(other.sigma / self.sigma) + (
                    self.sigma ** 2 + (self.mu - other.mu) ** 2) / other.sigma ** 2 - 1)

    def __add__(self, other):
        return Gaussian(mu=self.mu + other.mu, sigma=self.sigma + other.sigma)

    def __matmul__(self, other):
        return Gaussian(mu=other @ self.mu, sigma=other @ self.sigma @ other.T)

    def __mul__(self, other):
        return Gaussian(mu=self.mu * other, sigma=self.sigma * other)

    def __truediv__(self, other):
        return Gaussian(mu=self.mu / other, sigma=self.sigma / other)
