"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        samples = X_train.shape[0]
        dim = X_train.shape[1]
        self.w = np.random.rand(dim, self.n_class)
        for epoch in range(self.epochs):
            print(epoch)
            for x in range(samples):
                for c in range(self.n_class):
                    if c != y_train[x]:
                        # print(c, y_train[x])
                        # print(np.transpose(self.w)[c].shape, np.transpose(self.w)[y_train[x]].shape)
                        # print("Wrong: ", np.vdot(np.transpose(self.w)[c], X_train[x]), "Right: ", np.vdot(np.transpose(self.w)[y_train[x]], X_train[x]))
                        if np.dot(np.transpose(self.w)[c], X_train[x]) > np.dot(np.transpose(self.w)[y_train[x]], X_train[x]):
                            self.w[:, y_train[x]] = self.w[:, y_train[x]] + self.lr * X_train[x]
                            np.transpose(self.w)[c] = np.transpose(self.w)[c] - self.lr * X_train[x]
                            # print(self.lr * X_train[x], self.lr * X_train[x])


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
        # print(y_pred)
        return [np.argmax(i) for i in y_pred]