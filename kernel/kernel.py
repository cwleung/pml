from abc import ABC, abstractmethod
from math import sqrt

import numpy as np
from scipy.linalg import cholesky


class Kernel(ABC):
    """Base class for kernels."""

    @abstractmethod
    def __call__(self, X, Y):
        """Compute the kernel matrix for two sets of points X and Y."""
        pass

    def matrix(self, X):
        """Compute the kernel matrix for a given set of points."""
        N = X.shape[0]
        eta = 1e-8
        K = self(X, X)
        return K + eta * np.eye(N)

    # Cholesky decomposition
    def trig(self, X):
        """Compute the Cholesky decomposition of the kernel matrix."""
        return cholesky(self.matrix(X), lower=True)


class LinearKernel(Kernel):
    """Linear kernel with random bias."""

    def __init__(self):
        self.b = np.random.randn()

    def __call__(self, X, Y):
        return self.b + np.dot(X, Y.T)


class ExponentialKernel(Kernel):
    """Exponential kernel."""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X, Y):
        return np.exp(-np.abs(X[:, np.newaxis] - Y[np.newaxis, :]) / self.sigma)


class SquaredExponentialKernel(Kernel):
    """Squared Exponential kernel (also known as RBF kernel)."""

    def __init__(self, l=1.0, sigma_f=1.0):
        self.l = l
        self.sigma_f = sigma_f

    def __call__(self, X, Y):
        sqdist = np.sum(X ** 2, 1).reshape(-1, 1) + np.sum(Y ** 2, 1) - 2 * np.dot(X, Y.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)


class GaussianKernel(Kernel):
    """Gaussian kernel."""

    def __init__(self, tau, sigma):
        self.tau = tau
        self.sigma = sigma

    def __call__(self, X, Y):
        tau_exp = np.exp(self.tau) if np.isscalar(self.tau) else np.exp(self.tau[:, np.newaxis])
        sigma_exp = np.exp(self.sigma) if np.isscalar(self.sigma) else np.exp(self.sigma[:, np.newaxis])
        sq_dist = np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        return tau_exp * np.exp(-sq_dist / sigma_exp)


class PeriodicKernel(Kernel):
    """Periodic kernel."""

    def __init__(self, tau, sigma):
        self.tau = tau
        self.sigma = sigma

    def __call__(self, X, Y):
        return np.exp(self.tau * np.cos(np.sum(X[:, np.newaxis] - Y[np.newaxis, :], axis=-1) / self.sigma))


class Matern3Kernel(Kernel):
    """Matérn kernel with nu=3/2."""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X, Y):
        r = np.sqrt(np.sum((X[:, np.newaxis] - Y[np.newaxis, :]) ** 2, axis=-1))
        return (1 + sqrt(3) * r / self.sigma) * np.exp(-sqrt(3) * r / self.sigma)


class Matern5Kernel(Kernel):
    """Matérn kernel with nu=5/2."""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X, Y):
        r = np.sqrt(np.sum((X[:, np.newaxis] - Y[np.newaxis, :]) ** 2, axis=-1))
        return (1 + sqrt(5) * r / self.sigma + 5 * r ** 2 / (3 * self.sigma ** 2)) * np.exp(-sqrt(5) * r / self.sigma)
