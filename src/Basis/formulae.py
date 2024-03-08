import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sigm = sigmoid(x)
    return sigm * (1 - sigm)

