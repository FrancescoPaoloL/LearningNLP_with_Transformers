import numpy as np
from training_data import inputs, outputs
from formulae import sigmoid, sigmoid_derivative

class SimpleFFN():
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def train(self, training_inputs, training_outputs, training_iterations):
        for i in range(training_iterations):
            output = self.elaborate(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * sigmoid_derivative(output))
            self.synaptic_weights += adjustments
    
    def elaborate(self, inputs):
        inputs = inputs.astype(float)
        output = sigmoid(np.dot(inputs, self.synaptic_weights))
        return output
    
if __name__ == "__main__":
    ffn = SimpleFFN()
    ffn.train(inputs, outputs, 100000)

    print("Synaptic weights after training:")
    print(ffn.synaptic_weights)
    print("Output after training:")
    print(outputs)

    A = float(input("Input 1: "))
    B = float(input("Input 2: "))
    C = float(input("Input 3: "))

    print("New situation: input data = ", A, B, C)
    print("Output data:")
    print(ffn.elaborate(np.array([A, B, C])))