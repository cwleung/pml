import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from kernel.acq_function import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound, ThompsonSampling
from kernel.kernel import SquaredExponentialKernel
from models.svgp import SVGP


def objective_function(X):
    x = X.flatten()

    gaussian = 2 * np.exp(-(x - 1) ** 2)
    sinusoidal = np.sin(3 * x)
    decay = 0.1 * x ** 2

    result = gaussian + sinusoidal - decay

    return result.reshape(-1, 1)


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

        self.X_history = [self.X_sample.copy()]
        self.Y_history = [self.Y_sample.copy()]

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

            self.X_history.append(self.X_sample.copy())
            self.Y_history.append(self.Y_sample.copy())

            print(f"Iteration {i + 1}: f({X_next[0][0]:.3f}) = {Y_next[0][0]:.3f}")

        return self.X_sample, self.Y_sample

    def get_best(self):
        best_idx = np.argmax(self.Y_sample)
        return self.X_sample[best_idx], self.Y_sample[best_idx]

    def plot_optimization(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

        x_plot = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 1000).reshape(-1, 1)
        y_plot = self.objective_function(x_plot)

        ax1.plot(x_plot, y_plot, 'b-', label='Objective function', alpha=0.5)

        ax1.scatter(self.X_sample, self.Y_sample, c='red', s=50, label='Sampled points')

        if len(self.X_sample) > 0:
            self.gpr.fit(self.X_sample, self.Y_sample)
            mu, std = self.gpr.predict(x_plot)
            ax1.plot(x_plot, mu, 'g--', label='GPR mean')
            ax1.fill_between(x_plot.flatten(),
                             mu.flatten() - 2 * std.flatten(),
                             mu.flatten() + 2 * std.flatten(),
                             color='g', alpha=0.2, label='GPR uncertainty')

        ax1.set_xlabel('X')
        ax1.set_ylabel('f(X)')
        ax1.set_title('Objective Function and Samples')
        ax1.legend()

        # Optimization progress
        best_values = [np.max(Y) for Y in self.Y_history]
        ax2.plot(range(len(best_values)), best_values, 'b-', label='Best observed value')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Best observed value')
        ax2.set_title('Optimization Progress')
        ax2.legend()

        plt.tight_layout()
        plt.show()


bounds = np.array([[-3, 3]]).reshape(-1, 2)  # X bounds

gpr = SVGP(SquaredExponentialKernel(), num_inducing_points=5)
optimizer = BayesianOptimizer(gpr, objective_function, bounds, acquisition_function='UCB')
X_sample, Y_sample = optimizer.optimize(n_iter=100)

best_params, best_value = optimizer.get_best()
print(f"Best parameters: {best_params}")
print(f"Best value: {best_value}")

optimizer.plot_optimization()
