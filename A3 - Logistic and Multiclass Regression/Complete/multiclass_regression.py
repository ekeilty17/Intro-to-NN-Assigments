import torch
import torch.nn as nn

class MultiClassRegression(nn.Module):

    def __init__(self, num_inputs, num_classes):
        super(MultiClassRegression, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_inputs, num_classes)
            #nn.Softmax(dim=1)                       # This is applied by nn.CrossEntropLoss for numerical stability
        )

    """ ML Algorithms """
    def forward(self, I):
        return self.fc(I)
        