"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        #^ no need to change that because we would need X_shape for initialization
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N, D = X_train.shape
        self.w = np.random.rand(D)  # initialize a very small weight to train
        for epoch in range(self.epochs):
            # Print intermediate accuracy
            y_pred = X_train @ self.w
            temp = [-1 if i<0 else 1 for i in y_pred]
            print("Epoch", epoch, "Accuracy", np.sum(y_train == temp) / len(y_train) * 100)
            
            for j in range(N):
                self.w += (self.lr * self.sigmoid(-y_train[j] * (self.w.T @ X_train[j])) * y_train[j] * X_train[j])

                
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        N, D = X_test.shape
        y = np.zeros(N)
        for i in range(N):
            num = self.w.T @ X_test[i]
            if (num > self.threshold):
                y[i] = 1
            else:
                y[i] = -1               
        return y