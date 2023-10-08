from helpers.plots import plotFile



plotFile("datasets/xor.mat",
    hidden_layer_size=4,
    learning_rate=0.001,
    max_epochs=10000
)