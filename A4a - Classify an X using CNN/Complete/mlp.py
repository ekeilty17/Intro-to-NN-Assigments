import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_size, output_size, num_hidden_layers=1, hidden_size=9):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers 
        self.hidden_size = hidden_size if type(hidden_size) == list else [int(hidden_size)] * num_hidden_layers
        self.output_size = output_size

        layers = [self.input_size[0] * self.input_size[1]] + self.hidden_size

        # Creating hidden layers of neural network
        self.Hidden = nn.ModuleList()
        for l1, l2 in zip(layers[:-1], layers[1:]):
            self.Hidden.append( 
                nn.Sequential(
                    nn.Linear(l1, l2),
                    nn.ReLU()
                )
            )
        # create output layer
        self.last = nn.Sequential(
            nn.Linear(layers[-1], self.output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.reshape(-1, self.input_size[0] * self.input_size[1])
        for fc in self.Hidden:
            x = fc(x)
        return self.last(x)