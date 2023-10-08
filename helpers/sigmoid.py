import numpy as np

def sigmoid(x):
        #x = np.clip(x, -500, 500)#securing overflow
        return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
        return x * (1 - x)