import torch
import torch.nn as nn

class SingleNeuronClassifier(nn.Module):

    def __init__(self, num_inputs, actfunction="relu", lr=0.1):
        super(SingleNeuronClassifier, self).__init__()
        
        # hyper parameters
        self.set_actfunction(actfunction)
        self.lr = lr

        self.fc = nn.Sequential(
            nn.Linear(num_inputs, 1),       # W^T X + b
            self.g
        )

    """ Activation Functions """
    def set_actfunction(self, actfunction):
        if actfunction.lower() == "linear":
            self.g = lambda x: x
        elif actfunction.lower() == "relu":
            self.g = nn.ReLU()
        elif actfunction.lower() == "sigmoid":
            self.g = nn.Sigmoid()
        else:
            raise ValueError(f"Don't support the {actfunction} activation function")

    """ ML Algorithms """
    def forward(self, I):        
        return self.fc(I).squeeze()
        