## Linear classifiers
### Spring 2023 CS444 Assignment 1

[Assignment Page](https://slazebni.cs.illinois.edu/spring23/assignment1.html)

#### Description
In this assignment, I've implemented a simple linear classifier, and I've been tasked with implementing additional classifiers and applying them to two different datasets:

1. **Rice dataset**: It's a straightforward binary classification dataset where the labels are either 0 or 1, different from the -1 and 1 labels I've seen in our lectures. This means I might need to adjust the labels or adapt the parameter update rules accordingly.

2. **Fashion-MNIST**: This dataset involves classifying images of various fashion items into multiple categories, making it a multi-class classification problem.

The main objective of this assignment is to deepen my understanding of classic machine learning methods and to become more proficient with scientific computing tools in Python. Additionally, I'll gain valuable experience in hyperparameter tuning and learn how to properly split my data into training, validation, and test sets.

I've downloaded the starting code provided for this assignment, which likely includes templates and setup to help me get started with my classifier implementations.

Specifically, I'm responsible for implementing four classifiers, each in its own Python file:
- **Logistic regression** (in "logistic.py").
- **Perceptron** (in "perceptron.py").
- **Support Vector Machine (SVM)** (in "svm.py").
- **Softmax** (in "softmax.py").

One notable detail is that multi-class prediction using logistic regression can be challenging. Therefore, I only need to use logistic regression on the Rice dataset.

To guide me through the entire assignment, there's a top-level notebook titled "CS 444 Assignment-1.ipynb." This notebook provides step-by-step instructions. The format and structure of this assignment have been inspired by the Stanford CS231n assignments, and some data loading and instructions have been borrowed from there.

#### Top level file:
`./CS 444 Assignment-1.ipynb`

#### [Results](./zihengc2_yutongz7_mp1_report.pdf)