import torch
import torch.nn as nn

class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size, num_hidden_layers=1, hidden_size=64, actfunction="relu", output_size=1, seed=None, **kwargs):
        super(MultiLayerPerceptron, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError