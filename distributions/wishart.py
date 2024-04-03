import numpy as np

"""numpy implementation of Wishart distribution"""


class WishartDistribution:
    def __init__(self, df, scale):
        self.df = df
        self.scale = scale

    def sample(self):
        # https://en.wikipedia.org/wiki/Wishart_distribution#Random_number_generation
        dim = self.scale.shape[0]
        chol = np.linalg.cholesky(self.scale)
        samples = np.random.normal(size=(dim, dim))
        for i in range(dim):
            samples[i, i] = np.sqrt(np.random.chisquare(self.df - i))
        return chol @ samples @ samples.T @ chol.T

    def log_pdf(self, x):
        return -0.5 * (
                self.df * np.log(np.linalg.det(self.scale)) + self.df * x.shape[0] * np.log(2) +
                np.log(np.linalg.det(x)) + np.trace(np.linalg.solve(self.scale, x))
        )

    def __add__(self, other):
        return WishartDistribution(df=self.df + other.df, scale=self.scale + other.scale)

    def __matmul__(self, other):
        if other.shape[0] != self.scale.shape[0]:
            raise ValueError(f"Cannot multiply with matrix of shape {other.shape}")
        return WishartDistribution(df=self.df, scale=other @ self.scale @ other.T)

    def kl_divergence(self, other):
        d = self.scale.shape[0]
        delta = other.scale - self.scale

        chol_self = np.linalg.cholesky(self.scale)
        chol_other = np.linalg.cholesky(other.scale)
        inv_chol_self = np.linalg.inv(chol_self)

        tr_term = np.trace(np.dot(np.dot(inv_chol_self.T, chol_other), inv_chol_self))
        log_det_term = 2.0 * np.sum(np.log(np.diag(chol_other))) - 2.0 * np.sum(np.log(np.diag(chol_self)))

        return 0.5 * (log_det_term - d + tr_term + np.trace(np.dot(np.dot(delta, np.linalg.inv(other.scale)), delta.T)))

    def plot(self, n_samples=100):
        """Plot the Wishart distribution"""
        import matplotlib.pyplot as plt
        samples = np.array([self.sample() for _ in range(n_samples)])
        plt.scatter(samples[:, 0, 0], samples[:, 0, 1])
        plt.show()


if __name__ == '__main__':
    wishart = WishartDistribution(df=3, scale=np.array([[1, 0.5], [0.5, 1]]))
    wishart.plot()
    print(wishart.sample())
    print(wishart.log_pdf(np.array([[1, 0.5], [0.5, 1]])))
    print(wishart.kl_divergence(WishartDistribution(df=3, scale=np.array([[1, 0.5], [0.5, 1]]))))
