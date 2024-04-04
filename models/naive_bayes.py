import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class NaiveBayes:
    def __init__(self):
        self.priors = None
        self.likelihoods = None
        self.classes = None
        self.n_classes = None
        self.n_features = None

    @staticmethod
    def load_and_vectorize_data(filepath):
        """
        Loads the data from a CSV file and vectorizes the text column.

        :param filepath: Path to the CSV file
        :return: Tuple containing vectorized text data and labels
        """
        # Load the data
        data = pd.read_csv(filepath)

        # Check if the required columns are in the dataframe
        if 'text' not in data.columns or 'label' not in data.columns:
            raise ValueError("CSV file must contain 'text' and 'label' columns.")

            # Extract text data and labels
        texts = data['text'].values
        labels = data['label'].values

        # Vectorize the text data
        vectorizer = CountVectorizer()
        text_data_vectorized = vectorizer.fit_transform(texts)

        return text_data_vectorized.toarray(), labels

    def fit(self, X, y):
        """
        Fit the model given the data

        :param X: Input data (n_samples, n_features)
        :param y: Target data (n_samples,)
        :return: None
        """
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]

        self.priors = np.zeros(self.n_classes)
        self.likelihoods = []

        for i, c in enumerate(self.classes):
            # all samples with class c
            X_c = X[y == c]
            # calculate prior
            self.priors[i] = len(X_c) / len(X)
            # calculate likelihood for each feature
            class_likelihoods = []
            for j in range(self.n_features):
                # calculate mean and variance of feature j for class c
                mean = np.mean(X_c[:, j])
                var = np.var(X_c[:, j])
                class_likelihoods.append({'mean': mean, 'var': var})
            self.likelihoods.append(class_likelihoods)

    def predict(self, X):
        """
        Predict the class of each sample

        :param X: Input data (n_samples, n_features)
        :return: Predicted classes (n_samples,)
        """
        n_samples = X.shape[0]
        posteriors = np.zeros((n_samples, self.n_classes))

        for i, c in enumerate(self.classes):
            for j in range(self.n_features):
                mean = self.likelihoods[i][j]['mean']
                var = self.likelihoods[i][j]['var']
                # Calculate the log probability density of Gaussian distribution
                likelihood = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((X[:, j] - mean) ** 2) / var
                posteriors[:, i] += likelihood
            posteriors[:, i] += np.log(self.priors[i])

        return self.classes[np.argmax(posteriors, axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def plot_decision_boundary(self, X, y, resolution=0.01):
        """
        Plots the decision boundary for a 2D dataset.

        :param X: Input data (n_samples, 2)
        :param y: Target labels (n_samples,)
        :param resolution: The resolution of the meshgrid for plotting
        :return: None
        """
        if X.shape[1] != 2:
            raise ValueError("plot_decision_boundary only works for 2D datasets.")

        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = plt.cm.RdYlBu

        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))

        Z = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)

        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # Plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=colors[idx],
                        marker=markers[idx], label=cl, edgecolor='black')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(loc='upper left')
        plt.show()


if __name__ == '__main__':
    X, y = NaiveBayes.load_and_vectorize_data("../data/spam.csv")
    # Fit model
    priors = [0.5, 0.5]
    model = NaiveBayes()
    model.fit(X, y)

    # Evaluate model
    print(f"Accuracy: {model.score(X, y)}")

    # Plot priors and likelihoods
    model.plot_decision_boundary(X, y)
