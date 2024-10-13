import numpy as np
from scipy.linalg import solve_triangular

from kernel.kernel import Kernel, GaussianKernel
from models.base_model import SupBaseModel


class GaussianProcess(SupBaseModel):
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.X_train = None
        self.y_train = None
        self.K = None
        self.L = None
        self.alpha = None

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y).reshape(-1, 1)

        self.L = self.kernel.trig(self.X_train)
        self.alpha = solve_triangular(self.L.T, solve_triangular(self.L, self.y_train, lower=True))

    def predict(self, X):
        X = np.asarray(X)
        Ks = self.kernel(self.X_train, X)
        Kss = self.kernel.matrix(X)

        mu = Ks.T.dot(self.alpha).flatten()
        v = solve_triangular(self.L, Ks, lower=True)
        var = np.diag(Kss) - np.sum(v ** 2, axis=0)

        return mu, var


import matplotlib.pyplot as plt

if __name__ == '__main__':
    def f(X):
        return np.sin(X)


    X = np.random.uniform(-5, 5, 100).reshape(-1, 1)
    y = f(X)
    kernel = GaussianKernel(tau=0, sigma=1)
    gp = GaussianProcess(kernel=kernel)
    gp.fit(X, y)

    X_test = np.linspace(-5, 5, 100).reshape(-1, 1)
    y_pred, y_var = gp.predict(X_test)

    plt.figure(figsize=(10, 5))
    plt.plot(X_test, f(X_test), 'r:', label=r'$f(x) = \sin(x)$')
    plt.plot(X, y, 'r.', markersize=10, label='Observations')
    plt.plot(X_test, y_pred, 'b-', label='Prediction')
    plt.fill_between(X_test.flatten(), y_pred - 1.96 * np.sqrt(y_var), y_pred + 1.96 * np.sqrt(y_var), alpha=0.2)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-3, 3)
    plt.legend()
    plt.show()
