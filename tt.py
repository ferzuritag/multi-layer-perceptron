import numpy as np
from helpers.multi_layer_perceptron import trainMLP, classifyMLP

input_patterns = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
output_patterns = np.array([[-1],[1],[1],[-1]])

Wij,Wjk = trainMLP(input_patterns, output_patterns, hidden_layer_size=4,max_epochs=5000, min_error=0.01,learning_rate=0.001)
predictions, err = classifyMLP(input_patterns, output_patterns, Wij, Wjk)

print(err)

