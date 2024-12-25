from abc import ABC, abstractmethod

import numpy as np
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

    def diag(self, X):
        """Compute the diagonal of the kernel matrix."""
        return torch.diag(self.matrix(X))

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
    """Linear kernel with bias."""

    def __init__(self, c=1.0):  # Added variance parameter
        self.c = c  # Variance parameter
        self.b = torch.tensor(0.0)  # Fixed bias term

    def __call__(self, X, Y):
        # Convert inputs to torch tensors if they aren't already
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.float32)

        return self.c * torch.matmul(X, Y.T) + self.b


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
    def __init__(self, sigma, variance=1.0):
        self.sigma = sigma
        self.variance = variance

    def __call__(self, X, Y):
        r = torch.sqrt(torch.sum((X.unsqueeze(1) - Y.unsqueeze(0)) ** 2, dim=-1) + 1e-12)
        r = torch.clamp(r, min=1e-12)

        sqrt3 = torch.sqrt(torch.tensor(3.0))
        term = sqrt3 * r / self.sigma

        return self.variance * torch.exp(-term) * (1.0 + term)


class Matern5Kernel(Kernel):
    def __init__(self, sigma, variance=1.0):
        self.sigma = sigma
        self.variance = variance

    def __call__(self, X, Y):
        r = torch.sqrt(torch.sum((X.unsqueeze(1) - Y.unsqueeze(0)) ** 2, dim=-1) + 1e-12)
        r = torch.clamp(r, min=1e-12)

        sqrt5 = torch.sqrt(torch.tensor(5.0))
        term = sqrt5 * r / self.sigma
        term2 = term ** 2

        return self.variance * torch.exp(-term) * (1.0 + term + term2 / 3.0)


class ScaleKernel(Kernel):
    """Scale kernel - learns input scaling for any base kernel."""

    def __init__(self, base_kernel, input_scale=1.0, output_scale=1.0):
        self.base_kernel = base_kernel
        # Using log scale for numerical stability
        self.log_input_scale = torch.nn.Parameter(torch.tensor(np.log(input_scale), dtype=torch.float32))
        self.log_output_scale = torch.nn.Parameter(torch.tensor(np.log(output_scale), dtype=torch.float32))

    def __call__(self, X, Y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.float32)

        input_scale = torch.exp(self.log_input_scale)
        output_scale = torch.exp(self.log_output_scale)

        X_scaled = X * input_scale
        Y_scaled = Y * input_scale

        return output_scale * self.base_kernel(X_scaled, Y_scaled)
