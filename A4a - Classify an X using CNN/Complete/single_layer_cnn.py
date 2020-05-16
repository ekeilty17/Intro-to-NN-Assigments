import torch
import torch.nn as nn

class SingleLayerCNN(nn.Module):

    def __init__(self, kernel_size, num_kernels):
        super(SingleLayerCNN, self).__init__()

        self.kernel_size = kernel_size
        self.num_kernels = num_kernels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_kernels, kernel_size=kernel_size, stride=1, padding=0),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(kernel_size[0] * kernel_size[1] * self.num_kernels, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(-1, self.kernel_size[0] * self.kernel_size[1] * self.num_kernels)
        return self.fc(x)