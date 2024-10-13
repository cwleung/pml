import matplotlib.pyplot as plt
import torch

from kernel.kernel import Kernel, SquaredExponentialKernel
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
        self.X_train = X
        self.y_train = y

        self.L = self.kernel.trig(self.X_train)
        self.alpha = torch.triangular_solve(self.y_train.unsqueeze(1), self.L.t(), upper=True)
        self.alpha = torch.triangular_solve(self.alpha, self.L, upper=True).squeeze()

    def predict(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        Ks = self.kernel(self.X_train, X)
        Kss = self.kernel.matrix(X)

        mu = Ks.t().matmul(self.alpha).flatten()
        v = torch.triangular_solve(Ks, self.L, upper=True)
        var = torch.diag(Kss) - torch.sum(v ** 2, dim=0)

        return mu, var


if __name__ == '__main__':
    def f(X):
        return torch.sin(X)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.rand(100, 1, device=device) * 10 - 5
    y = f(X)
    kernel = SquaredExponentialKernel(l=1.0, sigma_f=1.0)
    gp = GaussianProcess(kernel=kernel)
    gp.fit(X, y)

    X_test = torch.linspace(-5, 5, 100, device=device).unsqueeze(1)
    y_pred, y_var = gp.predict(X_test)

    # Convert to numpy for plotting
    X = X.cpu().numpy()
    y = y.cpu().numpy()
    X_test = X_test.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_var = y_var.cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(X_test, f(torch.from_numpy(X_test)).cpu().numpy(), 'r:', label=r'$f(x) = \sin(x)$')
    plt.plot(X, y, 'r.', markersize=10, label='Observations')
    plt.plot(X_test, y_pred, 'b-', label='Prediction')
    plt.fill_between(X_test.flatten(), y_pred - 1.96 * torch.sqrt(torch.from_numpy(y_var)).numpy(),
                     y_pred + 1.96 * torch.sqrt(torch.from_numpy(y_var)).numpy(), alpha=0.2)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-3, 3)
    plt.legend()
    plt.show()