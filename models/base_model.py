from abc import ABC, abstractmethod


class SupBaseModel(ABC):

    @abstractmethod
    def fit(self, X, y):
        """Fit the model to the data"""

    @abstractmethod
    def predict(self, X):
        """Predict the target variable"""


class UnsupBaseModel(ABC):

    @abstractmethod
    def fit(self, X):
        """Fit the model to the data"""

    @abstractmethod
    def predict(self, X):
        """Predict the target variable"""
