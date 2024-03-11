import numpy as np
from training_data import inputs, outputs
from formulae import sigmoid, sigmoid_derivative

class SimpleFFN():
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def forwardPropagation(self, inputs):
        inputs = inputs.astype(float)
        output = sigmoid(np.dot(inputs, self.synaptic_weights))
        return output
    
    def backward(self, training_inputs, training_outputs, output):
        error = training_outputs - output
        adjustments = np.dot(training_inputs.T, error * sigmoid_derivative(output))
        return adjustments
    
    def train(self, training_inputs, training_outputs, learning_rate=0.1, epochs=1000):
        for epoch in range(epochs):
            output = self.forwardPropagation(training_inputs)
            
            error = training_outputs - output
            mse = np.mean(np.square(error))
            
            adjustments = self.backward(training_inputs, training_outputs, output)
            
            # Update weights
            self.synaptic_weights += adjustments * learning_rate
            
            # Print MSE for monitoring
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: MSE = {mse:.4f}')


if __name__ == "__main__":
    print("Original inputs:")
    print(inputs)
    print("Real outputs:")
    print(outputs)

    ffn = SimpleFFN()
    learning_rate = 0.1
    epochs = 1000
    ffn.train(inputs, outputs, learning_rate, epochs)

    print("Synaptic weights after training:")
    print(ffn.synaptic_weights)
    print("Output after training:")
    print(ffn.forwardPropagation(inputs))

    A = float(input("Input 1: "))
    B = float(input("Input 2: "))
    C = float(input("Input 3: "))

    print("New situation: input data = ", A, B, C)
    print("Output data:")
    print(ffn.forwardPropagation(np.array([A, B, C])))
