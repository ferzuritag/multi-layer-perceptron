import numpy as np
from helpers.sigmoid import sigmoid, sigmoid_derivative
import math
import matplotlib.pyplot as plt
import math

np.random.seed(52)
#X = patterns
#Y = outputs
#H = neurons on hidden layer
#nu = learning_rate
#Tmax = max epoch
#opt = draw learning curve

def trainMLP(patterns, outputs, n_neurons, learning_rate, max_epochs, min_error, opt):
    
    n_outputs = 1
    patterns = np.array(patterns)
    
    n_patterns, n_entries = patterns.shape
    biases = np.ones((n_patterns, 1))

    patterns = np.hstack((patterns,biases))
    n_patterns, n_entries = patterns.shape
     

    W_input_hidden = np.random.rand(n_entries, n_neurons)
    W_hidden_output = np.random.rand(n_neurons, n_outputs)

    mean_history = [] 
    mean_error = float("Inf")

    epoch_index = 0

    while((mean_error > min_error) & (epoch_index < max_epochs)):
        #forward
        hidden_output = sigmoid(np.dot(patterns, W_input_hidden))
        net_outputs = sigmoid(np.dot(hidden_output, W_hidden_output))
        
        #backpropagation
        output_error = net_outputs - outputs

        mean_error = np.mean(np.abs(output_error))
        mean_history.append(mean_error)

        delta_output = output_error * sigmoid_derivative(net_outputs)
        error_hidden = delta_output.dot(W_hidden_output.T)
        delta_hidden = error_hidden * sigmoid_derivative(hidden_output)

        #update weights
        W_hidden_output += hidden_output.T.dot(delta_output) * learning_rate
        W_input_hidden += patterns.T.dot(delta_hidden) * learning_rate

        epoch_index += 1

    return W_input_hidden, W_hidden_output

def classifMultiLayerPerceptron(patterns, outputs, W_input_hidden,W_hidden_output):
    patterns = np.array(patterns)
    
    n_patterns, n_entries = patterns.shape
    biases = np.ones((n_patterns, 1))

    patterns = np.hstack((patterns,biases))
    n_patterns, n_entries = patterns.shape

    hidden_output = sigmoid(np.dot(patterns, W_input_hidden))
    net_outputs = sigmoid(np.dot(hidden_output, W_hidden_output))

    return net_outputs


X = [[0,0],[0,1],[1,0],[1,1]]
Y = [[0],[1],[1],[0]]

Wij,Wjk = trainMLP(
    X,
    Y,
    learning_rate=0.0001,
    max_epochs=4000,
    min_error=math.pow(math.e, -9),
    n_neurons=3,
    opt=True
)

foo = classifMultiLayerPerceptron(X, Y, Wij, Wjk)
print(foo)
