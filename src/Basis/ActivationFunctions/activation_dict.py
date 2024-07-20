import numpy as np
from activation_functions import *

def get_activation_functions(x_values, x_softmax):
    return {
        "Step Function": step_function(x_values),
        "Piecewise Linear Function": piecewise_linear_function(x_values),
        "Parametric ReLU": parametric_relu(x_values),
        "Shifted ReLU": shifted_relu(x_values),
        "Maxout ReLU": maxout_relu(x_values),
        "SoftPlus Function": softplus(x_values),
        "Exponential ReLU": exponential_relu(x_values),
        "Swish ReLU": swish(x_values),
        "Sigmoid Function": sigmoid(x_values),
        "Softmax Function": softmax(x_softmax)
    }
