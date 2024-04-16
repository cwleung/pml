import numpy as np

from distributions.base import Distribution


class GammaDistribution(Distribution):

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __add__(self, other):
        return GammaDistribution(alpha=self.alpha + other.alpha, beta=self.beta + other.beta)

    def __matmul__(self, other):
        pass

    def log_pdf(self, x) -> np.ndarray:
        return (self.alpha * np.log(self.beta) - np.log(np.math.gamma(self.alpha)) +
                (self.alpha - 1) * np.log(x) - self.beta * x)

    def kl_divergence(self, other):
        return (self.alpha - other.alpha) * (np.log(other.beta) - np.log(self.beta)) + \
            (other.alpha - self.alpha) * (np.math.gamma(self.alpha) / np.math.gamma(other.alpha)) * \
            (self.beta / other.beta) ** self.alpha

    def sample(self, n_samples=1) -> np.ndarray:
        return np.random.gamma(shape=self.alpha, scale=1 / self.beta, size=n_samples)

    def plot(self, n_samples=100):
        import matplotlib.pyplot as plt
        samples = self.sample(n_samples)
        plt.hist(samples, bins=20, density=True)
        plt.title("Gamma Distribution")
        plt.show()


if __name__ == '__main__':
    gamma = GammaDistribution(alpha=2, beta=1)
    gamma.plot()
    print(gamma.sample(10))
    print(gamma.log_pdf(1))
    print(gamma.kl_divergence(GammaDistribution(alpha=2, beta=1)))
    print(gamma + GammaDistribution(alpha=2, beta=1))
