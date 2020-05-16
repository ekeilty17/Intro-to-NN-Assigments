import torch
import torch.nn as nn

class LogisticRegression(nn.Module):

    def __init__(self, num_inputs):
        super(LogisticRegression, self).__init__()

        raise NotImplementedError
        # recall, the nn.BCEWithLogitsLoss() activation function applies nn.Sigmoid() for you

    """ ML Algorithms """
    def forward(self, I):
        raise NotImplementedError