import torch
import torch.nn as nn

class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size, output_size=1):
        super(MultiLayerPerceptron, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
