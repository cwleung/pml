import numpy as np

from distributions.gaussian import MultivariateGaussian
from distributions.wishart import WishartDistribution


class NormalInvertedWishartDistribution:
    def __init__(self, mu, kappa, psi, nu):
        self.mu = mu
        self.kappa = kappa
        self.psi = psi
        self.nu = nu

        self.inv_wishart = WishartDistribution(df=nu, scale=psi)
        self.normal = MultivariateGaussian(mu=mu, sigma=psi / kappa)

    def sample(self):
        sigma = self.inv_wishart.sample()
        mu = self.normal.sample()
        return mu, sigma

    def marginal_likelihood(self, data):
        mu, sigma = data
        return self.normal.log_pdf(mu) + self.inv_wishart.log_pdf(sigma)

    def predict(self, data):
        mu, sigma = data
        return self.normal.sample(), self.inv_wishart.sample()

    def update(self, data):
        mu, sigma = data
        new_psi = self.psi + sigma
        new_kappa = self.kappa + 1
        new_nu = self.nu + 1
        new_mu = (self.kappa * self.mu + mu) / new_kappa
        return NormalInvertedWishartDistribution(new_mu, new_kappa, new_psi, new_nu)

    def kl_divergence(self, other):
        return self.normal.kl_divergence(other.normal) + self.inv_wishart.kl_divergence(other.inv_wishart)

    def plot(self, n_samples=100):
        self.normal.plot(n_samples)
        self.inv_wishart.plot(n_samples)


if __name__ == '__main__':
    niw = NormalInvertedWishartDistribution(mu=np.array([0, 0]), kappa=1, psi=np.array([[1, 0.5], [0.5, 1]]), nu=3)
    niw.plot()
    print(niw.sample())
    print(niw.marginal_likelihood((np.array([0, 0]), np.array([[1, 0.5], [0.5, 1]]))))
    print(niw.kl_divergence(NormalInvertedWishartDistribution(mu=np.array([0, 0]), kappa=1, psi=np.array([[1, 0.5], [0.5, 1]]), nu=3)))
    print(niw.update((np.array([0, 0]), np.array([[1, 0.5], [0.5, 1]]))))