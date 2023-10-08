import numpy as np
import h5py
import matplotlib.pyplot as plt
from helpers.multi_layer_perceptron import trainMLP, classifyMLP
from helpers.common import make_numpy_array_with_biases
import math


def plotFile(file_path, hidden_layer_size, learning_rate, max_epochs, min_error_for_convergence, draw_mean):
    file = h5py.File(file_path,'a')
    #fig, ax = plt.subplots()

    training_data_patterns = np.array(file.get("Xtr"))
    training_data_outputs = np.array(file.get("Ytr"))

    training_data_patterns = make_numpy_array_with_biases(training_data_patterns)

    W_input_hidden, W_hidden_output = trainMLP(
        training_data_patterns,
        training_data_outputs,
        hidden_layer_size=hidden_layer_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        min_error_for_convergence=min_error_for_convergence,
        opt=draw_mean
    )

    test_data_patterns = np.array(file.get("Xtt"))
    test_data_outputs = np.array(file.get("Ytt"))
    
    test_data_patterns = make_numpy_array_with_biases(test_data_patterns)

    _, output_errors = classifyMLP(test_data_patterns,test_data_outputs, W_input_hidden=W_input_hidden, W_hidden_output=W_hidden_output)

    bad_classified = 0
    good_classified = 0
    for index, pattern in enumerate(test_data_patterns):
        has_min_error = np.abs(output_errors[index]) < min_error_for_convergence
        color = "red" if test_data_outputs[index] == 1 else "blue"
        marker = "x" if not has_min_error else "."

        if(has_min_error):
            good_classified += 1
        else:
            bad_classified += 1
        plt.scatter(x=pattern[0],y=pattern[1], color=color, marker=marker)

        
    plt.xlabel(xlabel=f"H={hidden_layer_size} Tmax={max_epochs} nu={learning_rate} bad {bad_classified} good={good_classified}")
    plt.show()