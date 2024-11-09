import numpy as np
import torch
from torch.autograd import Variable

from kernel.kernel import Kernel, SquaredExponentialKernel, PeriodicKernel
from models.base_model import SupBaseModel


class SVGP(SupBaseModel):
    def __init__(self, kernel: Kernel, num_inducing_points: int = 10):
        self.kernel = kernel
        self.num_inducing_points = num_inducing_points
        self.Z = None
        self.m = None
        self.L = None
        self.noise_var = None  # Added observation noise variance

    def initialize_variational_parameters(self, X):
        # Initialize inducing points using k-means or uniform spacing
        if X.shape[0] > self.num_inducing_points:
            idx = np.linspace(0, X.shape[0] - 1, self.num_inducing_points, dtype=int)
            self.Z = torch.tensor(X[idx], requires_grad=True, dtype=torch.float32)
        else:
            self.Z = torch.tensor(X, requires_grad=True, dtype=torch.float32)

        # Initialize variational parameters
        self.m = Variable(torch.zeros((self.num_inducing_points, 1)), requires_grad=True)
        L = torch.eye(self.num_inducing_points) * 0.1  # Smaller initial variance
        self.L = Variable(L, requires_grad=True)
        self.noise_var = Variable(torch.tensor([0.1]), requires_grad=True)  # Initial noise variance

    def elbo(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        # Compute kernel matrices
        Kzz = self.kernel.matrix(self.Z) + torch.eye(self.num_inducing_points) * 1e-6  # Add jitter for stability
        Kzx = self.kernel(self.Z, X)
        Kxx_diag = self.kernel.diag(X)

        # Compute terms for ELBO
        L = torch.tril(self.L)  # Ensure L is lower triangular
        S = L @ L.t()

        # Stable computation of inverse using Cholesky
        L_Kzz = torch.linalg.cholesky(Kzz)
        alpha = torch.cholesky_solve(Kzx, L_Kzz)  # This computes Kzz^{-1} Kzx

        # Predictive distribution
        mu = (alpha.t() @ self.m).squeeze()
        var = Kxx_diag - torch.sum(Kzx * alpha, dim=0) + torch.sum((alpha.t() @ S) * alpha.t(), dim=1)
        var = var + self.noise_var

        # Log likelihood term
        log_lik = -0.5 * torch.sum((y.squeeze() - mu) ** 2 / var) \
                  - 0.5 * torch.sum(torch.log(var)) \
                  - 0.5 * y.shape[0] * np.log(2 * np.pi)

        # KL divergence term
        Kzz_inv = torch.cholesky_solve(torch.eye(self.num_inducing_points), L_Kzz)
        kl_div = 0.5 * (
                torch.trace(Kzz_inv @ S) +
                (self.m.t() @ Kzz_inv @ self.m) -
                self.num_inducing_points +
                torch.logdet(Kzz) -
                torch.logdet(S + torch.eye(self.num_inducing_points) * 1e-6)
        )

        return log_lik - kl_div

    def fit(self, X, y, learning_rate=0.01, num_iterations=2000):
        self.initialize_variational_parameters(X)
        optimizer = torch.optim.Adam([self.Z, self.m, self.L, self.noise_var], lr=learning_rate)

        for i in range(num_iterations):
            optimizer.zero_grad()
            loss = -self.elbo(X, y)
            loss.backward()
            optimizer.step()

            # Ensure positive noise variance
            with torch.no_grad():
                self.noise_var.data.clamp_(min=1e-6)

    def predict(self, X_test):
        X_test = torch.tensor(X_test, dtype=torch.float32)
        Kzz = self.kernel.matrix(self.Z) + torch.eye(self.num_inducing_points) * 1e-6
        Kzx = self.kernel(self.Z, X_test)
        Kxx = self.kernel.diag(X_test)

        L_Kzz = torch.linalg.cholesky(Kzz)
        alpha = torch.cholesky_solve(Kzx, L_Kzz)

        mu = (alpha.t() @ self.m).squeeze()
        L = torch.tril(self.L)
        S = L @ L.t()
        var = Kxx - torch.sum(Kzx * alpha, dim=0) + torch.sum((alpha.t() @ S) * alpha.t(), dim=1)
        var = var + self.noise_var

        return mu.detach().numpy(), var.detach().numpy()


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
