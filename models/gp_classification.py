import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.stats import norm

from data.dataset import generate_data


class GaussianProcessClassifier:
    def __init__(self, kernel, optimizer='L-BFGS-B', n_restarts=0, jitter=1e-8):
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts = n_restarts
        self.jitter = jitter

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.f = np.zeros(len(self.y_train))

        # Add jitter to the diagonal for numerical stability
        self.K = self.kernel(self.X_train, self.X_train)
        self.K += np.eye(len(self.X_train)) * self.jitter

        self._laplace_approximation()
        return self


    def _laplace_approximation(self, max_iter=10, tol=1e-6):
        for i in range(max_iter):
            p = self._sigmoid(self.f)
            W = np.maximum(p * (1 - p), 1e-10)  # Ensure W is positive
            grad = (self.y_train - p)

            # Compute B = I + W^(1/2) K W^(1/2)
            W_sqrt = np.sqrt(W)
            B = np.eye(len(self.y_train)) + (W_sqrt[:, None] * self.K * W_sqrt[None, :])

            # Add small diagonal term for numerical stability
            B += np.eye(len(self.y_train)) * self.jitter

            try:
                L = np.linalg.cholesky(B)
            except np.linalg.LinAlgError:
                print("Warning: Adding more jitter for numerical stability")
                B += np.eye(len(self.y_train)) * self.jitter * 10
                L = np.linalg.cholesky(B)

            b = W * self.f + grad
            a = b - W * np.linalg.solve(L.T, np.linalg.solve(L, self.K @ b))
            f_new = self.K @ a

            if np.max(np.abs(f_new - self.f)) < tol:
                break
            self.f = f_new

        self.W = W
        self.L = L

    def predict_proba(self, X):
        K_star = self.kernel(X, self.X_train)
        f_star_mean = K_star @ np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.K @ (self.y_train - 0.5)))

        # Compute predictive variance
        v = np.linalg.solve(self.L, K_star.T)
        f_star_var = self.kernel(X, X) + self.jitter - v.T @ v
        f_star_var = np.clip(f_star_var, 1e-10, np.inf)

        # Compute probabilities
        probas = self._sigmoid(f_star_mean / np.sqrt(1 + np.pi * f_star_var / 8))
        return np.vstack((1 - probas, probas)).T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def plot_decision_boundary(self, X, y, X_test=None, y_test=None, title="GP Classification", figsize=(10, 8)):
        """
        Plot the decision boundary, training points, and test predictions if available

        Parameters:
        -----------
        X : array-like of shape (n_samples, 2)
            Training data
        y : array-like of shape (n_samples,)
            Training target values
        X_test : array-like of shape (n_samples, 2), optional
            Test data for predictions
        y_test : array-like of shape (n_samples,), optional
            True test values for comparison
        title : str
            Title for the plot
        figsize : tuple
            Figure size (width, height)
        """
        if X.shape[1] != 2:
            raise ValueError("This visualization only works with 2D input data")

        # Create mesh grid
        margin = 0.5
        x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
        y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

        h = min(x_max - x_min, y_max - y_min) / 100
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h)
        )

        # Get predictions for mesh points
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        probas = self.predict_proba(mesh_points)
        Z = probas[:, 1].reshape(xx.shape)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot decision boundary with probability contours
        levels = np.linspace(0, 1, 20)
        contour = ax.contourf(xx, yy, Z, levels=levels, cmap='RdYlBu', alpha=0.7)

        # Add decision boundary line
        ax.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='--', linewidths=2)

        # Plot training points
        scatter_train = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                                   edgecolors='black', s=100, linewidth=1.5,
                                   norm=Normalize(vmin=0, vmax=1),
                                   label='Training points')

        # Plot test points if provided
        if X_test is not None and y_test is not None:
            scatter_test = ax.scatter(X_test[:, 0], X_test[:, 1],
                                      c=y_test, cmap='RdYlBu',
                                      marker='s', s=100,
                                      edgecolors='black', linewidth=1.5,
                                      norm=Normalize(vmin=0, vmax=1),
                                      label='Test points')

        ax.set_title(title, pad=20, fontsize=14)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)

        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper right')

        # Set consistent axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        return fig, ax


class RBFKernel:
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def __call__(self, X1, X2):
        dist_matrix = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * dist_matrix)


def get_probability_distribution_at_time(gpc, X_normalized, t, price_range=0.5, n_points=100):
    """
    Compute the probability distribution at a specific time t.

    Parameters:
    -----------
    gpc : GaussianProcessClassifier
        Fitted Gaussian Process Classifier
    X_normalized : array-like
        Normalized training data used for computing statistics
    t : float
        The time point at which to compute the distribution
    price_range : float
        Range of prices to consider (as a fraction of the mean price)
    n_points : int
        Number of points to evaluate

    Returns:
    --------
    dict containing:
        prices: array of price points
        probabilities: predicted probabilities
        confidence_intervals: tuple of (lower_bound, upper_bound)
    """
    # Normalize the time point
    t_normalized = (t - X_normalized.mean(axis=0)[0]) / X_normalized.std(axis=0)[0]

    # Create price range
    mean_price = X_normalized.mean(axis=0)[1]
    std_price = X_normalized.std(axis=0)[1]
    prices = np.linspace(mean_price - price_range, mean_price + price_range, n_points)

    # Create test points
    X_test = np.column_stack([np.full(n_points, t_normalized), prices])

    # Compute posterior distribution
    K_star = gpc.kernel(X_test, gpc.X_train)
    f_star_mean = K_star @ np.linalg.solve(gpc.L.T, np.linalg.solve(gpc.L, gpc.K @ (gpc.y_train - 0.5)))

    # Compute predictive variance
    v = np.linalg.solve(gpc.L, K_star.T)
    f_star_var = gpc.kernel(X_test, X_test) - v.T @ v
    f_star_var = np.diag(f_star_var)
    f_star_var = np.clip(f_star_var, 1e-10, np.inf)

    # Compute probabilities and confidence intervals
    kappa = 1.0 / np.sqrt(1 + np.pi * f_star_var / 8)
    prob_accept = gpc._sigmoid(f_star_mean * kappa)

    # 95% confidence intervals
    z = norm.ppf(0.975)  # 95% confidence level
    margin = z * np.sqrt(f_star_var)
    lower_bound = gpc._sigmoid(f_star_mean - margin)
    upper_bound = gpc._sigmoid(f_star_mean + margin)

    # Convert normalized prices back to original scale
    original_prices = prices * std_price + X_normalized.mean(axis=0)[1]

    return {
        'prices': original_prices,
        'probabilities': prob_accept,
        'confidence_intervals': (lower_bound, upper_bound)
    }


if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic data
    X, y, X_test, y_test = generate_data()
    prices_accept = X[y == 1, 1]
    prices_reject = X[y == 0, 1]

    # Normalize the features
    X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

    # Create and fit the model
    kernel = RBFKernel(gamma=2)
    gpc = GaussianProcessClassifier(kernel=kernel)
    gpc.fit(X_normalized, y)

    X_test_normalized = (X_test - X.mean(axis=0)) / X.std(axis=0)

    # Plot with both training and test data
    fig, ax = gpc.plot_decision_boundary(
        X_normalized, y,
        X_test=X_test_normalized if 'X_test_normalized' in locals() else None,
        y_test=y_test if 'y_test' in locals() else None,
        title="Decision Boundary for a seller accepting a given valuation"
    )
    plt.show()