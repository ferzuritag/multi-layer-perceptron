from helpers.plots import plotFile



plotFile("datasets/ring.mat",
    hidden_layer_size=8,
    learning_rate=0.001,
    max_epochs=1000
)