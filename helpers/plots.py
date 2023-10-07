import numpy as np
import h5py
import matplotlib.pyplot as plt
from main import trainMLP, classifyMLP
from itertools import combinations

def plotFile(file_path):
    file = h5py.File(file_path,'r')
    #fig, ax = plt.subplots()

    training_data_patterns = np.array(file.get("Xtr"))
    training_data_outputs = np.array(file.get("Ytr"))


    W_input_hidden, W_hidden_output = trainMLP(
        training_data_patterns,
        training_data_outputs,
        hidden_layer_size=3,
        learning_rate=0.1,
        max_epochs=100,
        min_error=0.1123412
    )

    test_data_patterns = np.array(file.get("Xtt"))
    test_data_outputs = np.array(file.get("Ytt"))

    max_on_x_axis = np.max(training_data_patterns[:, 0])
    min_on_x_axis = np.min(training_data_patterns[:, 0])\
    
    max_on_y_axis = np.max(training_data_patterns[:, 1])
    min_on_y_axis = np.min(training_data_patterns[:, 1])
    

    for index, pattern in enumerate(test_data_patterns):
        color = "blue" if test_data_outputs[index] == 0 else "red"
        plt.scatter(x=pattern[0],y=pattern[1], color=color, marker='x')

    steps= .05

    i = min_on_x_axis
    while(i < max_on_x_axis):
        j = min_on_y_axis
        while(j < max_on_y_axis):
            prediction, _ = classifyMLP(np.array([i,j]), outputs=[[1]],W_hidden_output=W_hidden_output, W_input_hidden=W_input_hidden)
            rounded_prediction = np.round(prediction)
            bg_color = "blue" if rounded_prediction == 0 else "red"
            plt.scatter(x=i,y=j, color=bg_color, marker='.', alpha=0.1)

            j += steps
        i += steps

    plt.show()