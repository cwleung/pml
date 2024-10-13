import numpy as np
from scipy.stats import norm


class AcquisitionFunction:
    def __init__(self, gp):
        self.gp = gp

    def __call__(self, X):
        raise NotImplementedError


class ExpectedImprovement(AcquisitionFunction):
    def __init__(self, gp, xi=0.01):
        super().__init__(gp)
        self.xi = xi

    def __call__(self, X):
        mu, sigma = self.gp.predict(X)
        mu_sample_opt = self.gp.Y.max()

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei


class ProbabilityOfImprovement(AcquisitionFunction):
    def __init__(self, gp, xi=0.01):
        super().__init__(gp)
        self.xi = xi

    def __call__(self, X):
        mu, sigma = self.gp.predict(X)
        mu_sample_opt = self.gp.Y.max()

        with np.errstate(divide='warn'):
            Z = (mu - mu_sample_opt - self.xi) / sigma
            pi = norm.cdf(Z)

        return pi


class UpperConfidenceBound(AcquisitionFunction):
    def __init__(self, gp, beta=2.0):
        super().__init__(gp)
        self.beta = beta

    def __call__(self, X):
        mu, sigma = self.gp.predict(X)
        return mu + self.beta * sigma


class ThompsonSampling(AcquisitionFunction):
    def __call__(self, X):
        return self.gp.sample(X)
