import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from distributions.categorical import Categorical
from distributions.normal import Gaussian
from models.base_model import SupBaseModel


class NaiveBayes(SupBaseModel):
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
        data = pd.read_csv(filepath)

        if 'text' not in data.columns or 'label' not in data.columns:
            raise ValueError("CSV file must contain 'text' and 'label' columns.")

        texts = data['text'].values
        labels = data['label'].values

        vectorizer = CountVectorizer()
        text_data_vectorized = vectorizer.fit_transform(texts)

        return text_data_vectorized.toarray(), labels

    def fit(self, X, y, alpha=1.0):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        self.priors = np.zeros(self.n_classes)
        self.likelihoods = [{} for _ in range(self.n_features)]

        n_categories = np.max(X, axis=0) + 1  # Determine number of categories for each feature
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[idx] = len(X_c) / len(X)

            for i in range(self.n_features):
                if np.all(np.equal(np.mod(X[:, i], 1), 0)):  # Is feature categorical?
                    self.likelihoods[i][c] = Categorical.mle(X_c[:, i], alpha=alpha, n_categories=n_categories[i])
                else:
                    self.likelihoods[i][c] = Gaussian.mle(X_c[:, i])

    def predict(self, X):
        """
        Predict the class labels for each sample in X.

        :param X: Input feature matrix (n_samples, n_features)
        :return: Array of predicted class labels (n_samples,)
        """
        n_samples = X.shape[0]
        posteriors = np.zeros((n_samples, self.n_classes))

        for i, c in enumerate(self.classes):
            log_prior = np.log(self.priors[i])  # log of prior probabilities

            log_likelihood = 0
            for j in range(self.n_features):
                feature_likelihoods = self.likelihoods[j][c]
                log_likelihood += feature_likelihoods.log_pdf(X[:, j])

            posteriors[:, i] = log_prior + log_likelihood

        predicted_class_indices = np.argmax(posteriors, axis=1)
        return self.classes[predicted_class_indices]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

        def plot_decision_boundary(self, X, y, resolution=0.01):
            """
            Plots the decision boundary for a 2D dataset.

            Parameters:
            X (numpy.ndarray): Input data (n_samples, n_features)
            y (numpy.ndarray): Target labels (n_samples,)
            features (list): The indices of the features to be used for the plot (default: [0, 1])
            resolution (float): The resolution of the meshgrid for plotting (default: 0.01)

            Returns:
            None
            """
            if X.shape[1] < 2:
                raise ValueError("plot_decision_boundary only works for 2D datasets.")
            if X.shape[1] > 2:
                from sklearn.manifold import TSNE
                X = TSNE(n_components=2).fit_transform(X)
                print("Finished t-SNE..., shape: ", X.shape)

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
    # X, y = NaiveBayes.load_and_vectorize_data("../data/spam.csv")
    # categorical data
    X = np.random.randint(0, 10, (1000, 2))
    y = np.where(np.dot(X, [1, 1]) > 5, 1, 0)
    priors = [0.5, 0.5]
    model = NaiveBayes()
    model.fit(X, y)
    print(f"Accuracy: {model.score(X, y)}")
    model.plot_decision_boundary(X, y)

    # do the continuous case
    X = np.random.rand(100, 2)
    y = np.where(np.dot(X, [1, 1]) > 1, 1, 0)
    model = NaiveBayes()
    model.fit(X, y)
    print(f"Accuracy: {model.score(X, y)}")
    model.plot_decision_boundary(X, y)

    # do mixture of both
    X = np.random.rand(100, 2)
    y = np.where(np.dot(X, [1, 1]) > 1, 1, 0)
    X = np.hstack((X, np.random.randint(0, 10, (100, 1))))
    print(X.shape)
    model = NaiveBayes()
    model.fit(X, y)
    print(f"Accuracy: {model.score(X, y)}")
    model.plot_decision_boundary(X, y)
