from distributions.base import Distribution
import numpy as np


class Categorical(Distribution):

    def __init__(self, p):
        self.p = p

    def __add__(self, other):
        return Categorical(p=self.p + other.p)

    def __matmul__(self, other):
        if other.shape[1] != 1:
            raise ValueError(f"Cannot multiply with matrix of shape {other.shape}")
        return Categorical(p=other @ self.p)

    def sample(self, n_samples=1):
        return np.random.choice(self.p, size=n_samples, p=self.p)

    def rsmaple(self, n_samples=1):
        return np.random.choice(self.p, size=n_samples, p=self.p)

    def kl_divergence(self, other):
        return np.sum(self.p * np.log(self.p / other.p))

    def log_pdf(self, x: np.array):
        return np.log(self.p[x])

    @staticmethod
    def mle(data: np.ndarray):
        """
        Maximum likelihood estimation of the Categorical distribution.
        :param data: Data samples (each row is a sample). (n_samples, n_features)
        :return: None (update the distribution)
        """
        # Compute the frequency of each category
        counts = np.bincount(data.squeeze(), minlength=len(np.unique(data)))

        # Normalize the counts to get the new probabilities
        p = counts / counts.sum()
        return Categorical(p=p)

    def plot(self):
        import matplotlib.pyplot as plt
        print(f"Distribution: {self.p}")
        plt.bar(range(len(self.p)), self.p)
        plt.show()


if __name__ == '__main__':
    cat = Categorical(p=np.array([0.1, 0.2, 0.3, 0.4]))
    cat.plot()
    print(cat.sample(10))
    print(cat.log_pdf([0, 1, 2, 3]))

    # example of mle
    Categorical.mle(data=np.array([[0], [0], [1], [1], [2], [2], [3], [3], [4]])).plot()
