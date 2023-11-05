## Multi-Layer Neural Networks
### Spring 2023 CS444 Assignment 2

[Assignment Page](https://slazebni.cs.illinois.edu/spring23/assignment2.html)

#### Description
In this assignment, I've been tasked with the implementation of multi-layer neural networks and the utilization of backpropagation to 'memorize' an image. As illustrated in the figure below, the inputs to the multi-layer network consist of 2-dimensional (x, y) coordinates, and the network's outputs are 3-dimensional RGB values. Essentially, the network's purpose is to reconstruct an image based on pixel coordinate inputs. My journey starts with training a model using the (x, y) coordinates directly, and later, I'll delve into different forms of input feature mapping, following the recommendations outlined in this paper, all in an effort to enhance the final image reconstruction.

The key tasks include writing my own forward and backward pass, training a four-layer network with both SGD and Adam optimizers, and ensuring that the network's weights are updated to minimize the mean square error (MSE) loss between the original and reconstructed images. Fortunately, I've been provided with certain hyperparameters, such as hidden layer size, learning rate, and activation function choices, which are detailed in the 'neural_network.ipnyb' notebook.

For setting up this assignment, I've downloaded the initial code from the provided link.

To guide me through all the necessary steps, there's a top-level notebook titled 'neural_network.ipynb.' This comprehensive notebook will provide step-by-step instructions on how to train a neural network for this specific task. My primary task is to implement the multi-layer neural network in the 'neural_net.py' file.

#### Top level file:
`./neural_network.ipynb`
#### [Results](./zihengc2_yutongz7_mp2_report.pdf)