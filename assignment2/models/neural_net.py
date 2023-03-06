"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str = "SGD"
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            
            if opt == "SGD":
                self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
                self.params["b" + str(i)] = np.zeros(sizes[i])
            
            elif opt == "Adam":
                self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
                self.params["b" + str(i)] = np.zeros(sizes[i])
                
                self.params["Mw" + str(i)] = np.zeros((sizes[i - 1], sizes[i]))
                self.params["Mb" + str(i)] = np.zeros(sizes[i])
                self.params["Vw" + str(i)] = np.zeros((sizes[i - 1], sizes[i]))
                self.params["Vb" + str(i)] = np.zeros(sizes[i])
            
        self.t = 0

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        return X @ W + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.maximum(0, X)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        return 1 * (X > 0)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_grad(self, y: np.ndarray) -> np.ndarray:
        # sigmoid_val = self.sigmoid(y)
        return y * (1 - y)

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
      # TODO implement this
      # (Observed - predicted)^2
    #   return np.square(np.subtract(y,p)).mean() * self.output_size
      return np.square(np.subtract(y,p)).mean()
    
    
    
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
      N = y.shape[0]
      return -2*(y - p)/N
      

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.
        self.outputs["Result" + str(0)] = X
        curr_X = X
        for curr_layer in range(1, self.num_layers + 1):
            curr_X = self.linear(self.params["W" + str(curr_layer)], curr_X, self.params["b" + str(curr_layer)])
            self.outputs["Linear" + str(curr_layer)] = curr_X
            
            # The network uses a ReLU nonlinearity after each fully connected layer
            if curr_layer < self.num_layers:
                curr_X = self.relu(curr_X)
                self.outputs["Result" + str(curr_layer)] = curr_X
             
            #  The outputs of the last fully-connected layer are passed through a sigmoid.
            else:
                curr_X = self.sigmoid(curr_X)
                self.outputs["Sigmoid" + str(curr_layer)] = curr_X
            
        return curr_X
    

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch ofinal_resultf training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.

        # Final result from forward
        final_pred = self.outputs["Sigmoid" + str(self.num_layers)]
        # print(final_pred.shape)

        # MSE Loss and gradient
        loss = self.mse(y, final_pred)
        # print("mse loss", loss)

        mse_grad = self.mse_grad(y, final_pred)
        # print("mse grad", mse_grad)

        # The final sigmoid layer
        grad = self.sigmoid_grad(final_pred) * mse_grad
        # print("Sigmoid grad", grad)
        self.gradients["Sigmoid" + str(self.num_layers)] = grad

        # Prior linear and relu layers
        for i in range(self.num_layers, 0, -1):
            W = self.params["W" + str(i)]
            b = self.params["b" + str(i)]
            
            # linear grad
            input_X = self.outputs["Result" + str(i - 1)]
            grad_W = input_X.T @ grad
            grad_b = np.ones(grad.shape[0]).T @ grad
            grad_X = grad @ W.T
            self.gradients["W" + str(i)] = grad_W / self.output_size
            self.gradients["b" + str(i)] = grad_b / self.output_size
            
            # relu grad
            if i > 1:
                input_X = self.outputs["Linear" + str(i - 1)]
                grad_Z = self.relu_grad(input_X)
                # print("Relu grad", grad_Z)
                grad = grad_X * grad_Z
        
        return loss


    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        if opt == "SGD":
            for i in range(1, self.num_layers + 1):
                self.params["W" + str(i)] -= lr * self.gradients["W" + str(i)]
                self.params["b" + str(i)] -= lr * self.gradients["b" + str(i)]
        elif opt == "Adam":
            self.t += 1
            for i in range(1, self.num_layers + 1):
                self.params["Mw" + str(i)] = b1 * self.params["Mw" + str(i)] + (1 - b1) * self.gradients["W" + str(i)]
                self.params["Mb" + str(i)] = b1 * self.params["Mb" + str(i)] + (1 - b1) * self.gradients["b" + str(i)]
                self.params["Vw" + str(i)] = b2 * self.params["Vw" + str(i)] + (1 - b2) * (self.gradients["W" + str(i)] ** 2)
                self.params["Vb" + str(i)] = b2 * self.params["Vb" + str(i)] + (1 - b2) * (self.gradients["b" + str(i)] ** 2)
                Mw_hat = self.params["Mw" + str(i)] / (1 - b1 ** self.t)
                Mb_hat = self.params["Mb" + str(i)] / (1 - b1 ** self.t)
                Vw_hat = self.params["Vw" + str(i)] / (1 - b2 ** self.t)
                Vb_hat = self.params["Vb" + str(i)] / (1 - b2 ** self.t)
                self.params["W" + str(i)] -= lr * Mw_hat / (Vw_hat ** 0.5 + eps)
                self.params["b" + str(i)] -= lr * Mb_hat / (Vb_hat ** 0.5 + eps)
        pass

