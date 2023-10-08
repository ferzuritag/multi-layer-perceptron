from helpers.plots import plotFile
import math

plotFile("datasets/linear.mat",
    hidden_layer_size=3,
    learning_rate=0.01,
    max_epochs=100000,
    min_error_for_convergence=math.pow(math.e,-9),
    draw_mean=True
)