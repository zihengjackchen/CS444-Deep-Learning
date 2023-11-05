## Self-supervised and transfer learning, object detection
### Spring 2023 CS444 Assignment 3

[Assignment Page](https://slazebni.cs.illinois.edu/spring23/assignment3.html)

#### Description
In this assignment, I've been given a two-part task aimed at helping me gain practical experience with PyTorch and the application of pre-trained deep learning models to new tasks. Let me describe it from my perspective:

Part 1: Self-supervised Learning on CIFAR10
In the first part, my goal is to utilize PyTorch to train a model for a self-supervised task, fine-tune a subset of the model's weights, and eventually train it in a fully supervised setting. I'll be working with the CIFAR10 dataset, which contains small (32x32) images categorized into 10 different object classes. For the self-supervised training, I'll ignore the provided class labels and focus on a task that involves rotating images randomly by 0, 90, 180, or 270 degrees. The network's objective is to classify the rotation angle of each input image using cross-entropy loss.

The model architecture I'll use is ResNet18, and I don't need to build it from scratch since there's a pre-implemented version in PyTorch. The steps include:

Train a ResNet18 on the rotation prediction task, generating rotated images and labels for this task, and report the test performance. The expected accuracy for rotation prediction on the test set is around 78%.
Fine-tune only the weights of the final block of convolutional layers and the linear layer on the supervised CIFAR10 classification task. Compare the performance between initializing from the rotation model and random weights, aiming for a test set accuracy of around 60%.
Train the entire network on the supervised CIFAR10 classification task, comparing the performance between initializing from the rotation model and random weights. The expected accuracy for the full pre-trained model is around 80%.
For extra credit, I can try to replicate a plot from a research paper that shows the advantages of pre-training on the Rotation task when only a small amount of labeled data is available. Additionally, I can experiment with more advanced models and attempt to train a rotation prediction model on the larger ImageNette dataset.

Part 2: YOLO Object Detection on PASCAL VOC
In the second part, my objective is to implement a YOLO-like object detector on the PASCAL VOC 2007 dataset. This detector aims to produce results similar to those shown in an example image. The main steps and considerations include:

Working with a provided pre-trained network structure for the model, inspired by DetNet, to implement the loss function of YOLO in the "yolo_loss.py" file.
While the network structure can be replaced by a different architecture and trained from scratch, the recommendation is to stick with the provided one for optimal accuracy with less computational expense.
Both parts of the assignment involve using PyTorch, and it's noted that Part 2 might require Google Colab Pro or Google Cloud Platform (GCP). The top-level notebooks associated with each part will provide guidance and steps for completing the tasks.

#### Top level file:
`./assignment3_part1/a3_part1_rotation.ipynb` and 
`./assignment3_part2/MP3_P2.ipynb`

#### [Results](./zihengc2_yutongz7_mp3_report.pdf)