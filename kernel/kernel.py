from abc import ABC, abstractmethod
from math import sqrt

import torch


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
        return K + eta * torch.eye(N, device=X.device)

    # Cholesky decomposition
    def trig(self, X):
        """Compute the Cholesky decomposition of the kernel matrix."""
        return torch.linalg.cholesky(self.matrix(X), upper=True)

    def __add__(self, other):
        return SumKernel(self, other)

    def __mul__(self, other):
        return ProductKernel(self, other)


class SumKernel(Kernel):
    """Sum of two kernels."""

    def __init__(self, kernel1, kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def __call__(self, X, Y):
        return self.kernel1(X, Y) + self.kernel2(X, Y)


class ProductKernel(Kernel):
    """Product of two kernels."""

    def __init__(self, kernel1, kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def __call__(self, X, Y):
        return self.kernel1(X, Y) * self.kernel2(X, Y)


class LinearKernel(Kernel):
    """Linear kernel with random bias."""

    def __init__(self):
        self.b = torch.randn(1)

    def __call__(self, X, Y):
        return self.b + torch.matmul(X, Y.T)


class ExponentialKernel(Kernel):
    """Exponential kernel."""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X, Y):
        return torch.exp(-torch.abs(X.unsqueeze(1) - Y.unsqueeze(0)) / self.sigma)


class SquaredExponentialKernel(Kernel):
    """Squared Exponential kernel (also known as RBF kernel)."""

    def __init__(self, l=1.0, sigma_f=1.0):
        self.l = l
        self.sigma_f = sigma_f

    def __call__(self, X, Y):
        sqdist = torch.sum(X ** 2, 1).view(-1, 1) + torch.sum(Y ** 2, 1) - 2 * torch.matmul(X, Y.T)
        return self.sigma_f ** 2 * torch.exp(-0.5 / self.l ** 2 * sqdist)


class PeriodicKernel(Kernel):
    """Periodic kernel."""

    def __init__(self, tau, sigma):
        self.tau = tau
        self.sigma = sigma

    def __call__(self, X, Y):
        return torch.exp(self.tau * torch.cos(torch.sum(X.unsqueeze(1) - Y.unsqueeze(0), dim=-1) / self.sigma))


class Matern3Kernel(Kernel):
    """Matérn kernel with nu=3/2."""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X, Y):
        r = torch.sqrt(torch.sum((X.unsqueeze(1) - Y.unsqueeze(0)) ** 2, dim=-1))
        return (1 + sqrt(3) * r / self.sigma) * torch.exp(-sqrt(3) * r / self.sigma)


class Matern5Kernel(Kernel):
    """Matérn kernel with nu=5/2."""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X, Y):
        r = torch.sqrt(torch.sum((X.unsqueeze(1) - Y.unsqueeze(0)) ** 2, dim=-1))
        return (1 + sqrt(5) * r / self.sigma + 5 * r ** 2 / (3 * self.sigma ** 2)) * torch.exp(
            -sqrt(5) * r / self.sigma)
