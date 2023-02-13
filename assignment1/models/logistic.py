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
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def helper(self, x):
        if x >= 0:
            z = np.exp(-z)
            return 1 / (1 + z)
        else:
            # if x is less than zero then z will be small, denom can't be
            # zero because it's 1+z.
            z = np.exp(z)
            return z / (1 + z)
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        return [self.helper(i) for i in z]

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        samples = X_train.shape[0]
        dim = X_train.shape[1]
        self.w = 0.1 * np.random.rand(dim, 1)
       
        # print(self.w.shape)
        
        for epoch in range(self.epochs):
            # print("Epoch", epoch, "w", self.w)
            for x_ind in range(samples):
                x = np.reshape(X_train[x_ind], (-1, 1))
                # print(x.shape)
                y = y_train[x_ind]
                # print((self.w.T @ x).shape)
                grad = self.sigmoid(-1 * y * self.w.T @ x) * y * x
                # print("Grad", grad.shape)
                self.w += self.lr * grad
        

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
        y_pred = X_test @ self.w
        print(y_pred)
        # print(y_pred)
        return [1 if i>self.threshold else 0 for i in y_pred]