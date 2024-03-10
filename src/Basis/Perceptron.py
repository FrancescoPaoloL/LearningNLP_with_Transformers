'''
+---------+--------+--------+
|         | inputs | output |
+---------+--------+--------+
| ex1     | 0 0 0  |      0 |
| ex2     | 1 1 1  |      1 |
| ex3     | 1 0 1  |      1 |
| ex4     | 0 1 1  |      0 |
|  ...    | .....  |      ..|
+---------+--------+--------+
'''

import numpy as np
from training_data import inputs, outputs
from formulae import *

if __name__ == "__main__":
    np.random.seed(1)
    synaptic_weights = 2 * np.random.random((3, 1)) - 1

    print("Original inputs:")
    print(inputs)
    print("Real outputs:")
    print(outputs)
    print("Random starting synaptic weights:")
    print(synaptic_weights)

    np.set_printoptions(suppress=True)

    for iteration in range(100000):
        # Forward propagation
        output = sigmoid(np.dot(inputs, synaptic_weights))

        # Backpropagation
        error = outputs - output
        adjustments = error * sigmoid_derivative(output)

        # Update weights
        synaptic_weights += np.dot(inputs.T, adjustments)

    print("Synaptic weights after training:")
    print(synaptic_weights)
    print("Output after training:")
    print(output)

