import torch
import torch.nn as nn

class SingleLayerCNN(nn.Module):

    def __init__(self, kernel_size, num_kernels, output_size=1):
        super(SingleLayerCNN, self).__init__()

        raise NotImplementedError

        self.conv = nn.Sequential(
            #nn.Conv2d(...)
        )

        self.fc = nn.Sequential(
            #nn.Linear(...)
        )

    def forward(self, x):
        raise NotImplementedError