import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class GaussianProcess:
    def __init__(self, kernel, noise=1e-8):
        self.kernel = kernel
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K = None
        self.L = None
        self.alpha = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        if y.ndim == 1:
            self.y_train = y.reshape(-1, 1)
        self.K = self.kernel(self.X_train, self.X_train)
        self.L = np.linalg.cholesky(self.K + self.noise * np.eye(len(self.X_train)))
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_train))

    def predict(self, X):
        Ks = self.kernel(self.X_train, X)
        Kss = self.kernel(X, X)

        mu = Ks.T.dot(self.alpha).reshape(-1)
        v = np.linalg.solve(self.L, Ks)
        var = np.diag(Kss - v.T.dot(v))

        return mu, var


def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    mu, sigma = gpr.predict(X)
    mu_sample = gpr.predict(X_sample)[0]

    sigma = sigma.reshape(-1, 1)

    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return -ei.ravel()  # Ensure output is 1D


def squared_exponential_kernel(X1, X2, l=1.0, sigma_f=1.0):
    sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * sqdist)


def objective_function(X):
    C = 10 ** X[:, 0]  # C parameter of SVM, log scale
    gamma = 10 ** X[:, 1]  # gamma parameter of SVM, log scale

    accuracies = []
    for c, g in zip(C, gamma):
        svm = SVC(C=c, gamma=g, random_state=42)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_val)
        accuracies.append(accuracy_score(y_val, y_pred))

    return np.array(accuracies).reshape(-1, 1)


def bayesian_optimization(f, bounds, n_iter, initial_points=5):
    dim = len(bounds)
    X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(initial_points, dim))
    Y_sample = f(X_sample)

    if Y_sample.ndim == 1:
        Y_sample = Y_sample.reshape(-1, 1)

    gpr = GaussianProcess(squared_exponential_kernel)

    X_history = [X_sample]
    Y_history = [Y_sample]

    for i in range(n_iter):
        gpr.fit(X_sample, Y_sample)

        result = minimize(
            lambda X: expected_improvement(X.reshape(1, -1), X_sample, Y_sample, gpr),
            x0=np.random.uniform(bounds[:, 0], bounds[:, 1], size=(1, dim)),
            bounds=bounds,
            method='L-BFGS-B'
        )

        X_next = result.x.reshape(1, -1)
        Y_next = f(X_next)

        if Y_next.ndim == 1:
            Y_next = Y_next.reshape(-1, 1)

        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))

        X_history.append(X_sample)
        Y_history.append(Y_sample)

        print(f"Iteration {i + 1}: f({X_next[0]}) = {Y_next[0][0]}")

    return X_sample, Y_sample, X_history, Y_history


def plot_optimization(bounds, X_history, Y_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    # Plot sampled points
    for i, X_sample in enumerate(X_history):
        if i == 0:
            ax1.scatter(X_sample[:, 0], X_sample[:, 1], c='r', s=50, label='Initial points')
        else:
            ax1.scatter(X_sample[-1, 0], X_sample[-1, 1], c='k', s=50, label='Sampled point' if i == 1 else "")

    ax1.set_xlabel('log10(C)')
    ax1.set_ylabel('log10(gamma)')
    ax1.set_title('Sampled Points in Parameter Space')
    ax1.legend()

    # Plot best observed value
    best_values = [np.max(Y) for Y in Y_history]
    ax2.plot(range(len(best_values)), best_values, 'b-', label='Best observed value')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best observed accuracy')
    ax2.set_title('Optimization Progress')
    ax2.legend()

    plt.tight_layout()
    plt.show()


# Load and prepare the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define the bounds for C and gamma (in log scale)
bounds = np.array([[-5, 5], [-5, 5]])  # log10(C) and log10(gamma)

# Run Bayesian optimization
X_sample, Y_sample, X_history, Y_history = bayesian_optimization(objective_function, bounds, n_iter=20)

# Find the best parameters
best_idx = np.argmax(Y_sample)
best_C = 10 ** X_sample[best_idx, 0]
best_gamma = 10 ** X_sample[best_idx, 1]
best_accuracy = Y_sample[best_idx, 0]

print(f"Best parameters: C={best_C:.4f}, gamma={best_gamma:.4f}")
print(f"Best validation accuracy: {best_accuracy:.4f}")

# Evaluate on test set
best_svm = SVC(C=best_C, gamma=best_gamma, random_state=42)
best_svm.fit(X_train, y_train)
test_accuracy = accuracy_score(y_test, best_svm.predict(X_test))
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot the optimization process
plot_optimization(bounds, X_history, Y_history)
