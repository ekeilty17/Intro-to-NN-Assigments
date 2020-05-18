import torch
import torch.nn as nn

class MultiLayerPerceptron(nn.Module):

    G = {
        "linear" : lambda x: x,
        "relu" : nn.ReLU(),
        "sigmoid" : nn.Sigmoid(),
        "tanh" : nn.Tanh()
    }

    def __init__(self, input_size, num_hidden_layers=1, hidden_size=64, actfunction="relu", output_size=1, seed=None, **kwargs):
        super(MultiLayerPerceptron, self).__init__()

        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers 
        self.hidden_size = hidden_size if type(hidden_size) == list else [int(hidden_size)] * num_hidden_layers
        self.actfuction = actfunction.lower()
        self.output_size = output_size

        layers = [self.input_size] + self.hidden_size

        # random seed
        self.seed = seed
        if self.seed != None:
            torch.manual_seed(self.seed)

        # Creating hidden layers of neural network
        self.Hidden = nn.ModuleList()
        for l1, l2 in zip(layers[:-1], layers[1:]):
            self.Hidden.append( 
                nn.Sequential(
                    nn.Linear(l1, l2),
                    self.G[self.actfuction]
                )
            )
        # create output layer
        self.last = nn.Sequential(
            nn.Linear(layers[-1], self.output_size),
            nn.Sigmoid() if self.output_size == 1 else nn.Softmax(dim=1)
            #nn.ReLU()
        )

    def forward(self, x):
        for fc in self.Hidden:
            x = fc(x)
        return self.last(x)