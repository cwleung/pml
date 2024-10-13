import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from kernel.acq_function import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound, ThompsonSampling
from kernel.kernel import SquaredExponentialKernel
from models.svgp import SVGP


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


class BayesianOptimizer:
    def __init__(
            self,
            gpr,
            objective_function,
            bounds,
            acquisition_function='EI'
    ):
        self.gpr = gpr
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.X_sample = None
        self.Y_sample = None
        self.X_history = []
        self.Y_history = []
        self.set_acquisition_function(acquisition_function)

    def set_acquisition_function(self, acquisition_function):
        if acquisition_function == 'EI':
            self.acquisition = ExpectedImprovement(self.gpr)
        elif acquisition_function == 'PI':
            self.acquisition = ProbabilityOfImprovement(self.gpr)
        elif acquisition_function == 'UCB':
            self.acquisition = UpperConfidenceBound(self.gpr)
        elif acquisition_function == 'TS':
            self.acquisition = ThompsonSampling(self.gpr)
        else:
            raise ValueError("Unknown acquisition function")

    def optimize(self, n_iter, initial_points=5):
        dim = len(self.bounds)
        self.X_sample = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(initial_points, dim))
        self.Y_sample = self.objective_function(self.X_sample)

        if self.Y_sample.ndim == 1:
            self.Y_sample = self.Y_sample.reshape(-1, 1)

        self.X_history = [self.X_sample]
        self.Y_history = [self.Y_sample]

        for i in range(n_iter):
            self.gpr.fit(self.X_sample, self.Y_sample)

            result = minimize(
                lambda X: -self.acquisition(X.reshape(1, -1)).ravel(),
                x0=np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(1, dim)),
                bounds=self.bounds,
                method='L-BFGS-B'
            )

            X_next = result.x.reshape(1, -1)
            Y_next = self.objective_function(X_next)

            if Y_next.ndim == 1:
                Y_next = Y_next.reshape(-1, 1)

            self.X_sample = np.vstack((self.X_sample, X_next))
            self.Y_sample = np.vstack((self.Y_sample, Y_next))

            self.X_history.append(self.X_sample)
            self.Y_history.append(self.Y_sample)

            print(f"Iteration {i + 1}: f({X_next[0]}) = {Y_next[0][0]}")

        return self.X_sample, self.Y_sample

    def get_best(self):
        best_idx = np.argmax(self.Y_sample)
        return self.X_sample[best_idx], self.Y_sample[best_idx]

    def plot_optimization(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

        for i, X_sample in enumerate(self.X_history):
            if i == 0:
                ax1.scatter(X_sample[:, 0], X_sample[:, 1], c='r', s=50, label='Initial points')
            else:
                ax1.scatter(X_sample[-1, 0], X_sample[-1, 1], c='k', s=50, label='Sampled point' if i == 1 else "")

        ax1.set_xlabel('log10(C)')
        ax1.set_ylabel('log10(gamma)')
        ax1.set_title('Sampled Points in Parameter Space')
        ax1.legend()

        best_values = [np.max(Y) for Y in self.Y_history]
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

# Try with a different acquisition function (e.g., Upper Confidence Bound)
gpr = SVGP(SquaredExponentialKernel(), num_inducing_points=5)
optimizer_ucb = BayesianOptimizer(gpr, objective_function, bounds, acquisition_function='UCB')
X_sample_ucb, Y_sample_ucb = optimizer_ucb.optimize(n_iter=20)

best_params_ucb, best_value_ucb = optimizer_ucb.get_best()
print(f"Best parameters (UCB): {best_params_ucb}")
print(f"Best value (UCB): {best_value_ucb}")

optimizer_ucb.plot_optimization()
