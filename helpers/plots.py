import numpy as np
import h5py
import matplotlib.pyplot as plt
from helpers.multi_layer_perceptron import trainMLP, classifyMLP
from helpers.common import make_numpy_array_with_biases
import math


def plotFile(file_path, hidden_layer_size, learning_rate, max_epochs):
    file = h5py.File(file_path,'r')
    #fig, ax = plt.subplots()

    training_data_patterns = np.array(file.get("Xtr"))
    training_data_outputs = np.array(file.get("Ytr"))

    training_data_patterns = make_numpy_array_with_biases(training_data_patterns)
    training_data_outputs =make_numpy_array_with_biases(training_data_outputs)


    W_input_hidden, W_hidden_output = trainMLP(
        training_data_patterns,
        training_data_outputs,
        hidden_layer_size=hidden_layer_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        min_error=math.pow(math.e, -1)
    )

    test_data_patterns = np.array(file.get("Xtt"))
    test_data_outputs = np.array(file.get("Ytt"))
    
    test_data_patterns = make_numpy_array_with_biases(test_data_patterns)
    test_data_outputs =make_numpy_array_with_biases(test_data_outputs)


    _, output_errors = classifyMLP(test_data_patterns,test_data_outputs, W_input_hidden=W_input_hidden, W_hidden_output=W_hidden_output)
    print(f"output_errors {output_errors}")
    #for index, pattern in enumerate(test_data_patterns):
    #    has_min_error = np.abs(output_errors[index]) < math.pow(math.e, -1)
    #    color = "red" if test_data_outputs[index] == 1 else "green" if not has_min_error else "blue"
    #    marker = "x" if not has_min_error else "."
    #    plt.scatter(x=pattern[0],y=pattern[1], color=color, marker=marker)

        
    plt.xlabel(xlabel=f"Usando {hidden_layer_size} neuronas y maximo {max_epochs} epocas")
    plt.show()