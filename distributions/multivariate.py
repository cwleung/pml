import numpy as np

from distributions.base import Distribution


class MultivariateGaussian(Distribution):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

        if len(mu.shape) != 1:
            raise ValueError(f"mu must be a vector, got shape: {mu.shape}")
        if len(sigma.shape) != 2:
            raise ValueError(f"sigma must be a matrix, got shape: {sigma.shape}")
        if mu.shape[0] != sigma.shape[0]:
            raise ValueError(f"mu and sigma must have the same dimension, got {mu.shape} and {sigma.shape}")
        self.dim = mu.shape[0]

    @property
    def precision(self):
        return np.linalg.inv(self.sigma)

    @property
    def L(self):
        return np.linalg.cholesky(self.sigma)

    @property
    def logdet(self):
        # |Sigma| = |L|^2
        # log |Sigma| = 2 log |L|
        return 2 * np.sum(np.log(np.diag(self.L)))

    @property
    def prec(self):
        return np.linalg.inv(self.sigma)

    def prec_mult(self, x):
        return np.linalg.cholesky(self.sigma) @ np.linalg.solve(self.L, x)

    def update(self, data: np.ndarray):
        """
        Update the Multivariate Gaussian distribution with new data samples.

        Args:
            data (numpy.ndarray): New data samples (each row is a sample).
        """
        n = data.shape[0]  # Number of samples
        mu_n = np.mean(data, axis=0)  # Sample mean
        sigma_n = np.cov(data.T, bias=True)  # Sample covariance

        # Update mean and covariance
        mu_new = (self.sigma @ mu_n + n * sigma_n @ self.mu) / (self.sigma + n * sigma_n)
        sigma_new = self.sigma @ sigma_n / (self.sigma + n * sigma_n)

        self.mu = mu_new
        self.sigma = sigma_new

    @staticmethod
    def mle(data: np.ndarray, alpha=1.0):
        """
        Maximum likelihood estimation of the Multivariate Gaussian distribution with Laplace smoothing.

        Args:
            data (numpy.ndarray): Data samples (each row is a sample).
            alpha (float): Laplace smoothing parameter (default is 1.0).
        """
        n, d = data.shape
        mu = (np.sum(data, axis=0) + alpha * d) / (n + alpha)
        centered_data = data - mu
        sigma = (np.dot(centered_data.T, centered_data) + alpha * np.eye(d)) / (n + alpha)
        return MultivariateGaussian(mu=mu, sigma=sigma)

    def __add__(self, other):
        return MultivariateGaussian(mu=self.mu + other.mu, sigma=self.sigma + other.sigma)

    def __matmul__(self, other):
        if other.shape[1] != self.dim:
            raise ValueError(f"Cannot multiply with matrix of shape {other.shape}")
        return MultivariateGaussian(mu=other @ self.mu, sigma=other @ self.sigma @ other.T)

    def sample(self, n_samples=1):
        return np.random.multivariate_normal(self.mu, self.sigma, size=n_samples)

    def log_pdf(self, x):
        # log-pdf of a Gaussian RV
        return (-0.5 * (x - self.mu) @ self.prec_mult(x - self.mu) -
                0.5 * self.logdet - 0.5 * len(self.mu) * np.log(2 * np.pi))

    def kl_divergence(self, other):
        d = self.dim
        delta_mu = self.mu - other.mu
        delta_mu = delta_mu.reshape(d, 1)
        delta_sigma = self.sigma - other.sigma
        inv_sigma = np.linalg.inv(self.sigma)
        inv_other_sigma = np.linalg.inv(other.sigma)
        tr_term = np.trace(np.dot(inv_other_sigma, self.sigma))
        delta_mu_t = delta_mu.T
        return 0.5 * (np.log(np.linalg.det(other.sigma) / np.linalg.det(self.sigma)) - d +
                      np.trace(np.dot(inv_other_sigma, self.sigma)) +
                      np.dot(np.dot(delta_mu.T, inv_other_sigma), delta_mu) +
                      np.trace(np.dot(np.dot(delta_sigma, inv_other_sigma), delta_sigma)))

    def plot(self, n_samples=1000, n_dim=2):
        """Plot the 2-d gaussian distribution"""
        import matplotlib.pyplot as plt
        samples = self.sample(n_samples)
        plt.scatter(samples[:, 0], samples[:, 1])
        plt.show()
