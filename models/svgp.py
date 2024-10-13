import numpy as np
import torch
from torch.autograd import Variable

from kernel.kernel import Kernel, SquaredExponentialKernel, PeriodicKernel
from models.base_model import SupBaseModel


class SVGP(SupBaseModel):
    def __init__(self, kernel: Kernel, num_inducing_points: int = 10):
        self.kernel = kernel
        self.num_inducing_points = num_inducing_points
        self.Z = None  # Inducing points
        self.m = None  # Variational mean
        self.S = None  # Variational covariance

    def initialize_variational_parameters(self, X):
        # Initialize inducing points
        idx = np.random.choice(X.shape[0], self.num_inducing_points, replace=False)
        self.Z = torch.tensor(X[idx], requires_grad=True, dtype=torch.float32)

        # Initialize variational parameters
        self.m = Variable(torch.zeros((self.num_inducing_points, 1)), requires_grad=True)
        L = torch.eye(self.num_inducing_points, requires_grad=True)
        self.L = Variable(L)

    def elbo(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        # Compute kernel matrices
        Kzz = self.kernel.matrix(self.Z)
        Kzx = self.kernel(self.Z, X)

        # Compute terms for ELBO
        Kzz_inv = torch.inverse(Kzz)
        S = self.L @ self.L.t()
        trace_term = torch.trace(Kzz_inv @ S)
        quad_term = self.m.t() @ Kzz_inv @ self.m

        # Compute predictive mean and variance
        mu = Kzx.t() @ Kzz_inv @ self.m
        var = torch.diag(self.kernel.matrix(X)) - torch.sum(Kzx.t() @ Kzz_inv @ Kzx, dim=1)

        # Compute log-likelihood
        log_lik = -0.5 * torch.sum((y - mu) ** 2 / var.reshape(-1, 1)) - 0.5 * torch.sum(torch.log(var))

        # Compute KL divergence
        kl_div = 0.5 * (trace_term + quad_term - self.num_inducing_points + torch.logdet(Kzz) - torch.logdet(S))

        return log_lik - kl_div

    def fit(self, X, y, learning_rate=0.003, num_iterations=1000):
        self.initialize_variational_parameters(X)
        optimizer = torch.optim.Adam([self.Z, self.m, self.L], lr=learning_rate)

        for _ in range(num_iterations):
            optimizer.zero_grad()
            loss = -self.elbo(X, y)  # Negative ELBO because we want to maximize it
            loss.backward()
            optimizer.step()

    def predict(self, X):
        """
        mu = K_{zx}^T K_{zz}^{-1} m
        var = K_{xx} - K_{zx}^T K_{zz}^{-1} K_{zx} + K_{zx}^T K_{zz}^{-1} S K_{zz}^{-1} K_{zx}
        """
        X = torch.tensor(X, dtype=torch.float32)
        Kzz = self.kernel.matrix(self.Z)
        Kzx = self.kernel(self.Z, X)

        Kzz_inv = torch.inverse(Kzz)
        mu = Kzx.t() @ Kzz_inv @ self.m
        S = self.L @ self.L.t()
        var = torch.diag(self.kernel.matrix(X)) - torch.sum(Kzx.t() @ Kzz_inv @ Kzx, dim=1) + torch.diag(
            Kzx.t() @ Kzz_inv @ S @ Kzz_inv @ Kzx)

        return mu.detach().numpy().flatten(), var.detach().numpy()


if __name__ == '__main__':
    # np.random.seed(42)
    n_samples = 100
    X = np.random.uniform(-3, 3, (n_samples, 1))
    y = np.sin(X) + np.random.randn(n_samples, 1) * 0.05

    kernel = PeriodicKernel(tau=1.0, sigma=1.0) * SquaredExponentialKernel()
    svgp = SVGP(kernel, num_inducing_points=10)
    svgp.fit(X, y)

    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_pred, y_var = svgp.predict(X_test)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Observations')
    plt.plot(X_test, y_pred, color='red', label='Predictive mean')
    plt.fill_between(X_test.flatten(), y_pred - 2 * np.sqrt(y_var), y_pred + 2 * np.sqrt(y_var),
                     color='red', alpha=0.2, label='±2 std dev')
    plt.legend()
    plt.show()
