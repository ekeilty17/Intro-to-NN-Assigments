import torch
import torch.nn as nn

class LogisticRegression(nn.Module):

    def __init__(self, num_inputs):
        super(LogisticRegression, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_inputs, 1)
            #nn.Sigmoid()               # This is applied by nn.BCEWithLogitsLoss for numerical stability
        )

    """ ML Algorithms """
    def forward(self, I):
        return self.fc(I)