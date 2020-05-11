import torch
import torch.nn as nn

class SingleLayerCNN(nn.Module):

    def __init__(self, kernel_size, num_kernels):
        super(SingleLayerCNN, self).__init__()

        raise NotImplementedError

        self.conv = nn.Sequential(

        )

        self.fc = nn.Sequential(
            
        )

    def forward(self, x):
        raise NotImplementedError