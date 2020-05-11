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

        # random seed
        self.seed = seed
        if self.seed != None:
            torch.manual_seed(self.seed)

        # Creating Hidden of neural network
        self.Hidden = nn.ModuleList()
        if self.num_hidden_layers == 0 or self.num_hidden_layers == []:
            self.last = nn.Sequential(
                nn.Linear(self.input_size, self.output_size),
                nn.Sigmoid() if output_size == 1 else nn.Softmax(dim=1)
                #nn.ReLU()
            )
        else:
            self.Hidden.append( 
                nn.Sequential(
                    nn.Linear(self.input_size, self.hidden_size[0]),
                    self.G[self.actfuction]
                )
            )
            for l1, l2 in zip(self.hidden_size[:-1], self.hidden_size[1:]):
                self.Hidden.append( 
                        nn.Sequential(
                            nn.Linear(l1, l2),
                            self.G[self.actfuction]
                        )
                    )
            self.last = nn.Sequential(
                nn.Linear(self.hidden_size[-1], self.output_size),
                nn.Sigmoid() if output_size == 1 else nn.Softmax(dim=1)
                #nn.ReLU()
            )

    def forward(self, x):
        for fc in self.Hidden:
            x = fc(x)
        return self.last(x)

if __name__ == "__main__":
    model = MultiLayerPerceptron(   103, 
                                    hidden_size=[64, 32], 
                                    output_size=1, 
                                    num_hidden_layers=2, 
                                    actfunction="relu", 
                                    seed=0
                                )
    
    print(model)