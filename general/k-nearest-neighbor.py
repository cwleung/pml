from collections import Counter

import numpy as np


def knn(X_train, y_train, X_test, k=3):
    y_pred = []
    for x in X_test:
        distances = np.linalg.norm(X_train - x, axis=1)
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_indices]

        label = Counter(nearest_labels).most_common(1)[0][0]
        y_pred.append(label)

    return np.array(y_pred)


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train the classifier on the training data
y_pred = knn(X_train, y_train, X_test, k=3)

# Compute the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
