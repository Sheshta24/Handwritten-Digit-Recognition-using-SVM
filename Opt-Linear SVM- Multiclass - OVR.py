"""
Created on Mon Mar 25 21:54:34 2024

@author: Sorna Raj S
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# Set random seed for reproducibility
np.random.seed(42)

start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.001, n_iters=2000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.losses = []
        self.accuracies = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.classifiers = {}

        for cls in self.classes:
            binary_y = np.where(y == cls, 1, -1)
            w, b = self._binary_fit(X, binary_y)
            self.classifiers[cls] = (w, b)

    def _binary_fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx in range(n_samples):
                x_i = X[idx]
                y_i = y[idx]
                condition = y_i * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_i))
                    self.b -= self.lr * y_i
            
            # Calculate loss and accuracy after each iteration
            loss = self._hinge_loss(X, y)
            acc = self._accuracy(X, y)
            self.losses.append(loss)
            self.accuracies.append(acc)

        return self.w, self.b

    def _hinge_loss(self, X, y):
        loss = 0.5 * np.dot(self.w, self.w) + self.lambda_param * np.maximum(0, 1 - y * (np.dot(X, self.w) - self.b)).sum()
        return loss

    def _accuracy(self, X, y):
        preds = np.sign(np.dot(X, self.w) - self.b)
        accuracy = np.mean(preds == y)
        return accuracy

    def predict(self, X):
        preds = []
        for x in X:
            class_preds = {cls: np.dot(x, w) - b for cls, (w, b) in self.classifiers.items()}
            preds.append(max(class_preds, key=class_preds.get))
        return np.array(preds)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)))
    for i in range(len(y_true)):
        matrix[y_true[i]][y_pred[i]] += 1
    return matrix

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_training_process(losses, accuracies):
    plt.figure(figsize=(12, 5))

    # Plot Loss vs steps
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(losses) + 1), losses, label='Loss', color='blue')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss vs Steps')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy vs steps
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(accuracies) + 1), accuracies, label='Accuracy', color='green')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Steps')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def select_subset(X, y, sample_size):
    indices = np.random.choice(len(X), sample_size, replace=False)
    return X[indices], y[indices]

# Load MNIST data
def load_data():
    train = pd.read_csv(r"C:\Users\91979\Downloads\mnist_train.csv", header=None)
    test = pd.read_csv(r"C:\Users\91979\Downloads\mnist_test.csv", header=None)

    X_train = train.drop(0, axis=1).to_numpy()
    y_train = train[0].to_numpy()
    X_test = test.drop(0, axis=1).to_numpy()
    y_test = test[0].to_numpy()

    # Feature scaling
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, y_train, X_test, y_test

# Main function
def main(sample_size):
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train = select_subset(X_train, y_train, sample_size)

    clf = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=2000)  # Increased number of iterations
    clf.fit(X_train, y_train)

    # Select a subset of samples from the testing data
    num_samples_to_show = 9
    sample_indices = np.random.choice(len(X_test), num_samples_to_show, replace=False)
    X_samples = X_test[sample_indices]
    y_samples_true = y_test[sample_indices]

    # Predict labels for the selected samples
    y_samples_pred = clf.predict(X_samples)

    # Display sample images along with predicted and actual labels
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        ax.imshow(X_samples[i].reshape(28, 28), cmap='gray')
        ax.set_title(f"True: {y_samples_true[i]}, Predicted: {y_samples_pred[i]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)

    train_acc = accuracy(y_train, train_preds)
    test_acc = accuracy(y_test, test_preds)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    
    end_time = dt.datetime.now() 
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time= end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)))

    # Plot confusion matrix for testing data
    plot_confusion_matrix(y_test, test_preds)

    # Plot training process
    plot_training_process(clf.losses, clf.accuracies)

# Sample size prompt
sample_size = int(input("Enter the sample size to train the SVM: "))
main(sample_size)
