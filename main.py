import numpy as np
import time
import math
from helpers.sigmoid import sigmoid, sigmoid_derivative


def trainMLP(
    input_patterns,
    patterns_output,
    hidden_layer_size,
    learning_rate,
    alpha,
    max_epochs,
    min_error_for_convergence,
    shouldGraphic = False,
    output_size = 1
):
    #set the patterns as numpy array
    patterns = np.array(input_patterns)
    #destructure rows and columns
    number_of_patterns, number_of_entries = patterns.shape
    #set the biases column
    biases_column = np.ones((number_of_patterns, 1), dtype=int)
    patterns = np.hstack((patterns,biases_column))
    #update the destructured columns
    number_of_patterns, number_of_entries = patterns.shape
    # initialize error counter
    error = float("inf")
    epoch_counter = 0
    #intialize weights
    weights_from_patterns_to_hidden_layer = np.random.rand(number_of_entries, hidden_layer_size)
    weights_from_hidden_layer_to_output = np.random.rand(hidden_layer_size, output_size)


    while ((error > min_error_for_convergence) & (epoch_counter < max_epochs)):
        time.sleep(1)
        print("for epoch", epoch_counter)
        for index,pattern in enumerate(patterns):
            #feed foward
            #calculate ponderated sum
            hidden_layer_sum = np.dot(pattern, weights_from_patterns_to_hidden_layer)
            #calculate output of hidden_layer
            hidden_layer_outputs = sigmoid(hidden_layer_sum) 
            #calculate the output
            output_sum = np.dot(hidden_layer_outputs, weights_from_hidden_layer_to_output)
            output = sigmoid(output_sum)

            loss = patterns_output[index] - output
            print(f"loss {loss}")

        epoch_counter += 1

patterns = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

patterns_output = [0,1,1,0]

trainMLP(input_patterns=patterns,
    patterns_output=patterns_output,
    hidden_layer_size=3,
    learning_rate=0.001,
    alpha=0.001,
    max_epochs=10000,
    min_error_for_convergence=math.pow(math.e, -9))