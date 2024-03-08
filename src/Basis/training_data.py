import numpy as np

# Define training data
inputs = np.array([[0, 0, 1],
                   [1, 1, 1],
                   [1, 0, 1],
                   [0, 1, 1],
                   [0, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0]])

outputs = np.array([[0, 1, 1, 0, 0, 1, 0]]).T
