#Luis Fernando Zurita Gonzalez

import numpy as np
from helpers.sigmoid import sigmoid, sigmoid_derivative
import matplotlib.pyplot as plt


np.random.seed(42)
#patterns and outputs should be np arrays
def classifyMLP(patterns,patterns_class,W_input_hidden,W_hidden_output):
    hidden_layer_w_sum = np.dot(patterns, W_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_w_sum)

    output_layer_sum = np.dot(hidden_layer_output, W_hidden_output)
    output_layer_output = sigmoid(output_layer_sum)

    error = patterns_class - output_layer_output
    
    return output_layer_output, error


def trainMLP(
    patterns,
    patterns_class,
    hidden_layer_size,
    learning_rate,
    max_epochs,
    min_error_for_convergence,
    opt= True
):
    output_size = 1
    _, n_of_entries = patterns.shape
    
    # Initialize weights with random values
    W_input_hidden = np.random.uniform(size=(n_of_entries, hidden_layer_size))
    W_hidden_output = np.random.uniform(size=(hidden_layer_size, output_size))
    mean_error_history = []
    mean_error = float("Inf")

    for epoch_index in range(max_epochs):
        # Feedforward
        hidden_layer_w_sum = np.dot(patterns, W_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_w_sum)

        output_layer_sum = np.dot(hidden_layer_output, W_hidden_output)
        output_layer_output = sigmoid(output_layer_sum)

        error = patterns_class - output_layer_output

        # Backpropagation
        d_output = error * sigmoid_derivative(output_layer_output)
        error_hidden = d_output.dot(W_hidden_output.T)
        d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

        # Update weights
        W_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
        W_input_hidden += patterns.T.dot(d_hidden) * learning_rate

        mean_error = np.mean(np.square(error))
        mean_error_history.append(mean_error)

        if (mean_error < min_error_for_convergence):
            print(f"Solved in {epoch_index}")
            break
        
    if(opt):
        plt.plot(np.arange(0, epoch_index + 1), mean_error_history)
        plt.xlabel(xlabel=f"Curva de aprendizaje H={hidden_layer_size} Tmax={max_epochs} nu={learning_rate}")
        plt.show()
        
    

    return W_input_hidden, W_hidden_output