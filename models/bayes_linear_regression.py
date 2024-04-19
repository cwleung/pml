import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from distributions.multivariate import MultivariateGaussian


def generate_sin_data_n_features(n_samples=100, n_features=2, noise=0.1):
    X = np.linspace(0, 10, n_samples)
    y = np.sin(X)
    if n_features > 1:
        X = np.vstack([X ** i for i in range(1, n_features + 1)]).T
    else:
        X = X.reshape(-1, 1)
    y += noise * np.random.randn(n_samples)
    return X, y


class BayesianLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.w_ = None

        self.prior = MultivariateGaussian(np.zeros(1), np.eye(1))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        A = self.alpha * np.eye(n_features) + self.beta * X.T @ X
        b = self.beta * X.T @ y
        self.w_ = np.linalg.solve(A, b)
        return self

    def predict(self, X, return_std=False):
        y_pred = X @ self.w_
        if return_std:
            y_var = np.sum((X @ np.linalg.inv(self.alpha * np.eye(X.shape[1]) + self.beta * X.T @ X)) * X, axis=1)
            y_std = np.sqrt(y_var)
            return y_pred, y_std
        return y_pred


if __name__ == '__main__':
    X, y = generate_sin_data_n_features(n_samples=100, n_features=4, noise=0.1)
    model = BayesianLinearRegression(alpha=1.0, beta=1.0)
    model.fit(X, y)
    y_pred, y_std = model.predict(X, return_std=True)

    plt.figure(figsize=(10, 5))
    plt.plot(X[:, 0], y, 'o', label='Data')
    plt.plot(X[:, 0], y_pred, label='Prediction')
    plt.fill_between(X[:, 0], y_pred - y_std, y_pred + y_std, color='gray', alpha=0.2, label='Uncertainty')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Bayesian Linear Regression')
    plt.legend()
    plt.show()
