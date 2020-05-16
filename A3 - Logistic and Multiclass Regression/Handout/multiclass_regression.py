import torch
import torch.nn as nn

class MultiClassRegression(nn.Module):

    def __init__(self, num_inputs, num_classes):
        super(MultiClassRegression, self).__init__()

        raise NotImplementedError
        # recall, the nn.CrossEntropyLoss() activation function applies nn.Softmax() for you

    """ ML Algorithms """
    def forward(self, I):
        raise NotImplementedError
        