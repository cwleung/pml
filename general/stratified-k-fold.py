import numpy as np


class LinearRegressionSGD:
    def __init__(self, learning_rate=0.01, epochs=1000, regularization='none', lambda_param=0.01):
        self.lr = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None

    def initialize_parameters(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def l1_regularization(self):
        return self.lambda_param * np.sign(self.weights)

    def l2_regularization(self):
        return self.lambda_param * self.weights

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.initialize_parameters(n_features)

        for _ in range(self.epochs):
            for idx in range(n_samples):
                x_i = X[idx]
                y_i = y[idx]

                y_pred = np.dot(x_i, self.weights) + self.bias
                error = y_pred - y_i

                if self.regularization == 'l1':
                    reg_term = self.l1_regularization()
                elif self.regularization == 'l2':
                    reg_term = self.l2_regularization()
                else:
                    reg_term = 0

                self.weights -= self.lr * (error * x_i + reg_term)
                self.bias -= self.lr * error

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# Example usage for L1 regularization
model_l1 = LinearRegressionSGD(regularization='l1', lambda_param=0.01)
X_train = np.random.randn(100, 3)
y_train = np.random.randn(100)
model_l1.fit(X_train, y_train)

# Example usage for L2 regularization
model_l2 = LinearRegressionSGD(regularization='l2', lambda_param=0.01)
model_l2.fit(X_train, y_train)
