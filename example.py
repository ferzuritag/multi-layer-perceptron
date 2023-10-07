import numpy as np

# Define the activation function and its derivative
class ActivationFunction:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

# Define the Multi-Layer Perceptron class
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with random values
        self.weights_input_hidden = np.random.uniform(size=(self.input_size, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(size=(self.hidden_size, self.output_size))
        
    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden)
        self.hidden_layer_output = ActivationFunction.sigmoid(self.hidden_layer_input)
        
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.output_layer_output = ActivationFunction.sigmoid(self.output_layer_input)
        return self.output_layer_output
        
    def backward(self, X, y, learning_rate):
        # Calculate the error
        error = y - self.output_layer_output

        # Backpropagation
        d_output = error * ActivationFunction.sigmoid_derivative(self.output_layer_output)
        error_hidden = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden * ActivationFunction.sigmoid_derivative(self.hidden_layer_output)

        # Update weights
        self.weights_hidden_output += self.hidden_layer_output.T.dot(d_output) * learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden) * learning_rate
        
    def train(self, X, y, num_iterations, learning_rate):
        for i in range(num_iterations):
            # Forward propagation
            self.forward(X)

            # Backpropagation and weight updates
            self.backward(X, y, learning_rate)

    def predict(self, X):
        return self.forward(X)

# Training data (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create an MLP with 2 input neurons, 4 hidden neurons, and 1 output neuron
mlp = MLP(input_size=2, hidden_size=4, output_size=1)

# Train the MLP
mlp.train(X, y, num_iterations=10000, learning_rate=0.1)

# Testing the trained network
predictions = mlp.predict(X)
print("Predicted Output:")
print(np.round(predictions))