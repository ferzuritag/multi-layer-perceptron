import numpy as np
from helpers.sigmoid import sigmoid, sigmoid_derivative


#patterns and outputs should be np arrays
def classifyMLP(patterns,outputs,W_input_hidden,W_hidden_output):
    hidden_layer_w_sum = np.dot(patterns, W_input_hidden)
    hidden_layer_output = sigmoid_derivative(hidden_layer_w_sum)

    output_layer_sum = np.dot(hidden_layer_output, W_hidden_output)
    output_layer_output = sigmoid(output_layer_sum)

    error = output_layer_output - outputs
    
    return output_layer_output, error


def trainMLP(
    patterns,
    outputs,
    hidden_layer_size,
    learning_rate,
    max_epochs,
    min_error
):
    output_size = 1
    _, n_of_entries = patterns.shape
    W_input_hidden = np.random.rand(n_of_entries, hidden_layer_size)
    W_hidden_output = np.random.rand(hidden_layer_size, output_size)

    epoch_index = 0
    mean_error = float("Inf")

    while((epoch_index < max_epochs)):
        #feed foward
        hidden_layer_w_sum = np.dot(patterns, W_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_w_sum)

        output_layer_sum = np.dot(hidden_layer_output, W_hidden_output)
        output_layer_output = sigmoid(output_layer_sum)

        error = output_layer_output - outputs
        # Backpropagation
        d_output = error * sigmoid_derivative(output_layer_output)
        error_hidden = d_output.dot(W_hidden_output.T)
        d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

        #update weights
        W_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
        W_input_hidden += patterns.T.dot(d_hidden) * learning_rate

        epoch_index += 1

    return W_input_hidden, W_hidden_output



X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

W_input_hidden, W_hidden_output = trainMLP(
    X,
    Y,
    hidden_layer_size=3,
    learning_rate=0.1,
    max_epochs=10000,
    min_error=0.1123412
)

predictions, _ = classifyMLP(X,Y,W_input_hidden, W_hidden_output)