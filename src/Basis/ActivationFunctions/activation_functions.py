import numpy as np

def step_function(x):
    return np.where(x >= 0, 1, 0)

def piecewise_linear_function(x):
    return np.where(x < -1, -0.5, np.where(x < 1, x, 0.5))

def parametric_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def shifted_relu(x, shift=1):
    return np.maximum(x - shift, 0)

def maxout_relu(x):
    return np.maximum(x, 0.5 * x)

def softplus(x):
    return np.log(1 + np.exp(x))

def exponential_relu(x):
    return np.where(x > 0, x, np.exp(x) - 1)

def swish(x, beta=1):
    return x / (1.0 + np.exp(-beta * x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
