from abc import ABC, abstractmethod

import numpy as np


class Distribution(ABC):
    @abstractmethod
    def __add__(self, other):
        """Addition of two distributions"""

    @abstractmethod
    def __matmul__(self, other):
        """Dot product of two distributions"""

    @abstractmethod
    def log_pdf(self, x) -> np.ndarray:
        """Log PDF of the distribution"""

    @classmethod
    def pdf(self, x: np.ndarray):
        return np.exp(self.log_pdf(x))

    # TODO: Implement entropy
    def entropy(self):
        """Entropy of the distribution"""

    @abstractmethod
    def kl_divergence(self, other):
        """KL divergence between two distributions"""

    @abstractmethod
    def sample(self, n_samples=1) -> np.ndarray:
        """Sample from the distribution"""

    def plot(self):
        raise NotImplementedError("Plotting not implemented for this distribution")
