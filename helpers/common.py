import numpy as np

def make_numpy_array_with_biases(array):
    biases_col = np.ones((array.shape[0], 1))
    return np.hstack((array, biases_col))