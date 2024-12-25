import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        samples, features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0

        for _ in range(self.iterations):
            predictions = self._sigmoid(np.dot(X, self.weights) + self.bias)
            # Update weights and bias
            self.weights -= self.lr * np.dot(X.T, (predictions - y)) / samples
            self.bias -= self.lr * np.mean(predictions - y)

    def predict(self, X):
        predictions = self._sigmoid(np.dot(X, self.weights) + self.bias)
        return np.round(predictions)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# Example usage
X = np.array([[1, 2, 3, 4, 5]]).T
y = np.array([0, 0, 1, 1, 1])

model = LogisticRegression(learning_rate=0.01, iterations=1000)
model.fit(X, y)
predictions = model.predict(X)
print("Predictions:", predictions)
