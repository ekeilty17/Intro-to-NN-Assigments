import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, num_classes):
        super(CNN, self).__init__()
        
        raise NotImplementedError

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=, out_channels=, kernel_size=, stride=, padding=),
            nn.MaxPool2d(kernel_size=)
        )
        self.conv2 = nn.Sequential(
            
        )
        self.fc = nn.Sequential(
            
        )

        # implement the archtecture LeNet from the assignment

    def forward(self, x):
        # x.size() = torch.Size([opts.batch_size, 1, 28, 28])
        # 1 = channel size (not 3 because they are black and white pictures
        # 28 = height and width of images in pixels
        raise NotImplementedError

        # Hint: you might have to reshape x
        # google torch.reshape()
