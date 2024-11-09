import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from data.dataset import generate_data


class GaussianProcessClassifier:
    def __init__(self, kernel, optimizer='L-BFGS-B', n_restarts=0):
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts = n_restarts

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.f = np.zeros(len(self.y_train))
        self.K = self.kernel(self.X_train, self.X_train)
        self._laplace_approximation()
        return self

    def _laplace_approximation(self, max_iter=10, tol=1e-6):
        for i in range(max_iter):
            p = self._sigmoid(self.f)
            W = p * (1 - p)
            grad = (self.y_train - p)
            hess = -np.diag(W)
            B = np.eye(len(self.y_train)) + W[:, np.newaxis] * self.K
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
        v = np.linalg.solve(self.L, K_star.T)
        f_star_var = self.kernel(X, X) - v.T @ v
        f_star_var = np.clip(f_star_var, 1e-10, np.inf)

        probas = self._sigmoid(f_star_mean / np.sqrt(1 + np.pi * f_star_var / 8))
        return np.vstack((1 - probas, probas)).T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def plot_decision_boundary(self, X, y, title="GP Classification", figsize=(15, 6)):
        """
        Plot the decision boundary, training points, and uncertainty regions with improved visualization

        Parameters:
        -----------
        X : array-like of shape (n_samples, 2)
            Training data (must be 2D for visualization)
        y : array-like of shape (n_samples,)
            Target values
        title : str
            Title for the plot
        figsize : tuple
            Figure size (width, height)
        """
        if X.shape[1] != 2:
            raise ValueError("This visualization only works with 2D input data")

        # Create a mesh grid with adaptive resolution
        margin = 0.5
        x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
        y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

        # Adjust mesh resolution based on data range
        range_x = x_max - x_min
        range_y = y_max - y_min
        h = min(range_x, range_y) / 100  # adaptive step size

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h)
        )

        # Get predictions for all mesh points
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        probas = self.predict_proba(mesh_points)
        uncertainty = -np.sum(probas * np.log(probas + 1e-10), axis=1)
        Z = probas[:, 1].reshape(xx.shape)
        uncertainty = uncertainty.reshape(xx.shape)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, y=1.05)

        # Plot decision boundary
        levels = np.linspace(0, 1, 20)
        contour = ax1.contourf(xx, yy, Z, levels=levels, cmap='RdYlBu', alpha=0.8)
        decision_boundary = ax1.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='--', linewidths=2)

        # Plot training points with larger markers and better contrast
        scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                              edgecolors='black', s=100, linewidth=1.5,
                              norm=Normalize(vmin=0, vmax=1))

        # Add colorbar for decision boundary
        plt.colorbar(contour, ax=ax1, label='Probability of Class 1')

        ax1.set_title('Decision Boundary', pad=20)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')

        # Plot uncertainty with improved colormap
        uncertainty_levels = np.linspace(uncertainty.min(), uncertainty.max(), 20)
        uncertainty_plot = ax2.contourf(xx, yy, uncertainty, levels=uncertainty_levels,
                                        cmap='viridis', alpha=0.8)

        # Plot training points on uncertainty plot
        ax2.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                    edgecolors='black', s=100, linewidth=1.5,
                    norm=Normalize(vmin=0, vmax=1))

        # Add colorbar for uncertainty
        plt.colorbar(uncertainty_plot, ax=ax2, label='Predictive Uncertainty (Entropy)')

        ax2.set_title('Predictive Uncertainty', pad=20)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Price')

        # Set consistent axis limits and aspect ratio
        for ax in [ax1, ax2]:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        return fig, (ax1, ax2)


class RBFKernel:
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def __call__(self, X1, X2):
        dist_matrix = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * dist_matrix)


if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic data
    X, y, X_test, y_test = generate_data()
    prices_accept = X[y == 1, 1]
    prices_reject = X[y == 0, 1]

    # Normalize the features
    X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

    # Create and fit the model
    kernel = RBFKernel(gamma=0.5)
    gpc = GaussianProcessClassifier(kernel=kernel)
    gpc.fit(X_normalized, y)

    # Plot both normalized and unnormalized data
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 12))

    # Plot unnormalized data
    ax1.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Accepted', alpha=0.6)
    ax1.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Rejected', alpha=0.6)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.set_title('Original Price-Time Data')
    ax1.legend()
    ax1.grid(True)

    # Plot price distributions
    bins = 30
    ax2.hist(prices_accept, bins=bins, alpha=0.5, color='blue', label='Accepted')
    ax2.hist(prices_reject, bins=bins, alpha=0.5, color='red', label='Rejected')
    ax2.set_xlabel('Price')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Price Distribution')
    ax2.legend()
    ax2.grid(True)

    # Plot GP classification results
    gpc.plot_decision_boundary(X_normalized, y, "GP Classification on Normalized Data")
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nAccepted Prices:")
    print(f"Mean: {np.mean(prices_accept):.2f}")
    print(f"Std Dev: {np.std(prices_accept):.2f}")
    print(f"Min: {np.min(prices_accept):.2f}")
    print(f"Max: {np.max(prices_accept):.2f}")

    print("\nRejected Prices:")
    print(f"Mean: {np.mean(prices_reject):.2f}")
    print(f"Std Dev: {np.std(prices_reject):.2f}")
    print(f"Min: {np.min(prices_reject):.2f}")
    print(f"Max: {np.max(prices_reject):.2f}")
