from helpers.plots import plotFile
import math 

plotFile("datasets/ads.mat",
    hidden_layer_size=6,
    learning_rate=.1,
    max_epochs=100000,
    draw_mean=True,
    min_error_for_convergence=math.pow(math.e, -9)
)