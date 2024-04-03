import numpy as np

from distributions.multivariate import MultivariateGaussian

import numpy as np
from scipy.stats import multivariate_normal


class NaiveBayes:
    def __init__(self):
        self.priors = None
        self.likelihoods = None

    def fit(self, X, y):
        """
        Fit the model given the data

        :param X: Input data (n_samples, n_features)
        :param y: Target data (n_samples,)
        :return: None
        """
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]

        self.priors = np.zeros(self.n_classes)
        self.likelihoods = np.empty((self.n_classes, self.n_features), dtype=object)

        for i, c in enumerate(self.classes):
            # all samples with class c
            X_c = X[y == c]
            # calculate prior
            self.priors[i] = len(X_c) / len(X)
            # calculate likelihood
            for j in range(self.n_features):
                # fit a univariate Gaussian to each feature
                self.likelihoods[i][j] = multivariate_normal(mean=np.mean(X_c[:, j]), cov=np.var(X_c[:, j]))

    def predict(self, X):
        """
        Predict the class of each sample

        :param X: Input data (n_samples, n_features)
        :return: Predicted classes (n_samples,)
        """
        n_samples = X.shape[0]
        posteriors = np.zeros((n_samples, self.n_classes))

        for i, c in enumerate(self.classes):
            likelihood = np.array([self.likelihoods[i][j].logpdf(X[:, j]) for j in range(self.n_features)]).sum(axis=0)
            posteriors[:, i] = np.log(self.priors[i]) + likelihood

        return self.classes[np.argmax(posteriors, axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def plot(self):
        import matplotlib.pyplot as plt
        plt.bar(range(len(self.priors)), self.priors)
        plt.show()
        for i, c in enumerate(self.classes):
            print(f"Class: {c}")
            for j in range(self.n_features):
                print(f"Feature {j}: {self.likelihoods[i][j].mean()}")


if __name__ == '__main__':
    # Mock data
    np.random.seed(0)
    n = 100
    X = np.random.randn(n, 2)
    y = np.random.randint(0, 2, n)

    # Fit model
    priors = [0.5, 0.5]
    # for each class
    likelihoods = [MultivariateGaussian(mu=np.mean(X[y == c], axis=0), sigma=np.cov(X[y == c].T)) for c in np.unique(y)]
    model = NaiveBayes()
    model.fit(X, y)

    # Evaluate model
    print(f"Accuracy: {model.score(X, y)}")

    # Plot priors and likelihoods
    model.plot()
