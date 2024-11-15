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
    if X.ndim == 1:
        X = X.reshape(1, -1)

    C = 10 ** X[:, 0]  # C parameter of SVM, log scale
    gamma = 10 ** X[:, 1]  # gamma parameter of SVM, log scale

    accuracies = []
    for c, g in zip(C, gamma):
        svm = SVC(C=c, gamma=g, random_state=42)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_val)
        accuracies.append(accuracy_score(y_val, y_pred))

    return np.array(accuracies)

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
        self.best_values = []
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

        # Initialize with random points
        self.X_sample = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(initial_points, dim)
        )
        self.Y_sample = self.objective_function(self.X_sample).reshape(-1, 1)

        self.best_values = [np.max(self.Y_sample)]
        self.X_history = [self.X_sample.copy()]
        self.Y_history = [self.Y_sample.copy()]

        for i in range(n_iter):
            # Fit GP
            self.gpr.fit(self.X_sample, self.Y_sample)

            # Find next point to evaluate
            x_next = self._propose_next_point()
            y_next = self.objective_function(x_next).reshape(-1, 1)

            # Update samples
            self.X_sample = np.vstack((self.X_sample, x_next))
            self.Y_sample = np.vstack((self.Y_sample, y_next))

            # Update history
            self.X_history.append(self.X_sample.copy())
            self.Y_history.append(self.Y_sample.copy())
            self.best_values.append(np.max(self.Y_sample))

            print(f"Iteration {i + 1}: f({x_next[0]}) = {y_next[0][0]:.4f}")

        return self.X_sample, self.Y_sample

    def _propose_next_point(self):
        dim = len(self.bounds)

        def objective(x):
            return -self.acquisition(x.reshape(1, -1)).ravel()[0]

        best_x = None
        best_acquisition_value = np.inf

        # Try multiple random starting points
        n_random_starts = 5
        for _ in range(n_random_starts):
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=dim)
            result = minimize(
                objective,
                x0=x0,
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            if result.fun < best_acquisition_value:
                best_acquisition_value = result.fun
                best_x = result.x

        return best_x.reshape(1, -1)

    def get_best(self):
        best_idx = np.argmax(self.Y_sample)
        return self.X_sample[best_idx], self.Y_sample[best_idx]

    def plot_optimization(self, resolution=50):
        """
        Enhanced visualization of the optimization process with multiple subplots:
        1. Contour plot of the objective function with optimization trajectory
        2. Acquisition function surface at the final iteration
        3. Convergence plot
        4. Parameter evolution over iterations
        """
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(2, 2)

        # 1. Contour plot of the objective function
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_objective_surface(ax1, resolution)

        # 2. Acquisition function surface
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_acquisition_surface(ax2, resolution)

        # 3. Convergence plot
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_convergence(ax3)

        # 4. Parameter evolution
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_parameter_evolution(ax4)

        plt.tight_layout()
        plt.show()

    def _plot_objective_surface(self, ax, resolution):
        """Plot the objective function surface with optimization trajectory."""
        # Create grid of points
        x = np.linspace(self.bounds[0, 0], self.bounds[0, 1], resolution)
        y = np.linspace(self.bounds[1, 0], self.bounds[1, 1], resolution)
        X, Y = np.meshgrid(x, y)

        # Evaluate objective function on grid
        Z = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = self.objective_function(np.array([[X[i, j], Y[i, j]]]))

        # Plot contour
        contour = ax.contour(X, Y, Z, levels=20)
        ax.clabel(contour, inline=True, fontsize=8)

        # Plot optimization trajectory
        initial_points = self.X_history[0]
        ax.scatter(initial_points[:, 0], initial_points[:, 1],
                   c='red', s=100, label='Initial points')

        # Plot subsequent points with color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.X_history) - 1))
        for i, X_sample in enumerate(self.X_history[1:]):
            ax.scatter(X_sample[-1, 0], X_sample[-1, 1],
                       c=[colors[i]], s=100, label=f'Iteration {i + 1}')

            # Draw arrows to show optimization path
            if i > 0:
                prev_point = self.X_history[i][-1]
                curr_point = X_sample[-1]
                ax.arrow(prev_point[0], prev_point[1],
                         curr_point[0] - prev_point[0],
                         curr_point[1] - prev_point[1],
                         head_width=0.1, head_length=0.1, fc='k', ec='k', alpha=0.3)

        ax.set_xlabel('log10(C)')
        ax.set_ylabel('log10(gamma)')
        ax.set_title('Objective Function Surface\nwith Optimization Trajectory')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    def _plot_acquisition_surface(self, ax, resolution):
        """Plot the acquisition function surface at the final iteration."""
        x = np.linspace(self.bounds[0, 0], self.bounds[0, 1], resolution)
        y = np.linspace(self.bounds[1, 0], self.bounds[1, 1], resolution)
        X, Y = np.meshgrid(x, y)

        # Evaluate acquisition function on grid
        Z = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = self.acquisition(np.array([[X[i, j], Y[i, j]]]))

        contour = ax.contour(X, Y, Z, levels=20)
        ax.clabel(contour, inline=True, fontsize=8)

        # Plot current best point
        best_x, _ = self.get_best()
        ax.scatter(best_x[0], best_x[1], c='red', s=200, marker='*',
                   label='Current best')

        ax.set_xlabel('log10(C)')
        ax.set_ylabel('log10(gamma)')
        ax.set_title('Acquisition Function Surface\nat Final Iteration')
        ax.legend()

    def _plot_convergence(self, ax):
        """Plot the convergence of the best observed value."""
        ax.plot(range(len(self.best_values)), self.best_values,
                'b-', marker='o', label='Best value')
        ax.fill_between(range(len(self.best_values)),
                        self.best_values, alpha=0.2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best observed value')
        ax.set_title('Optimization Convergence')
        ax.grid(True)
        ax.legend()

    def _plot_parameter_evolution(self, ax):
        """Plot the evolution of parameters over iterations."""
        iterations = range(len(self.X_history))

        # Extract parameters
        c_values = [X[-1, 0] for X in self.X_history]
        gamma_values = [X[-1, 1] for X in self.X_history]

        # Plot both parameters
        ax.plot(iterations, c_values, 'b-', marker='o', label='log10(C)')
        ax.plot(iterations, gamma_values, 'r-', marker='o', label='log10(gamma)')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter value')
        ax.set_title('Parameter Evolution')
        ax.grid(True)
        ax.legend()

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

# Initialize and run optimization
gpr = SVGP(SquaredExponentialKernel(), num_inducing_points=5)
optimizer_ucb = BayesianOptimizer(gpr, objective_function, bounds, acquisition_function='UCB')
X_sample_ucb, Y_sample_ucb = optimizer_ucb.optimize(n_iter=20)

best_params_ucb, best_value_ucb = optimizer_ucb.get_best()
print(f"Best parameters (UCB): C=10^{best_params_ucb[0]:.3f}, gamma=10^{best_params_ucb[1]:.3f}")
print(f"Best accuracy (UCB): {best_value_ucb[0]:.4f}")

optimizer_ucb.plot_optimization()