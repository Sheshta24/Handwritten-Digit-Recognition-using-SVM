
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:52:07 2024

@author: sheshta
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
mnist = fetch_openml('mnist_784', version=1, parser='auto')

# Extract data and labels
X, y = mnist['data'], mnist['target']
y = y.astype(np.int8)  # Convert target to integer

# Splitting the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# It's usually a good idea to scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
X_test_scaled = scaler.transform(X_test.astype(np.float64))

# Custom SVM Classifier
class LinearSVM:
    def __init__(self, C=1, max_iter=1000, eta=0.001):
        self.C = C
        self.max_iter = max_iter
        self.eta = eta
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.max_iter):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.eta * (2 * self.w / n_samples)
                else:
                    self.w += self.eta * (x_i * y_[idx] - 2 * self.w / n_samples)
                    self.b += self.eta * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# One-vs-Rest approach to handle multi-class
class OvRClassifier:
    def __init__(self, classifier, *args, **kwargs):
        self.classifier = classifier
        self.classifiers = []
        self.args = args
        self.kwargs = kwargs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for class_ in self.classes_:
            y_binary = np.where(y == class_, 1, -1)
            clf = self.classifier(*self.args, **self.kwargs)
            clf.fit(X, y_binary)
            self.classifiers.append(clf)

    def predict(self, X):
        predictions = np.array([clf.predict(X) for clf in self.classifiers]).T
        return np.argmax(predictions, axis=1)

# Train custom SVM
ovr_clf = OvRClassifier(LinearSVM)
ovr_clf.fit(X_train_scaled, y_train)  # Using a subset for faster training

# Test on a subset of the test set
test_predictions = ovr_clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, test_predictions)

print("Accuracy on the testing subset:", accuracy)
