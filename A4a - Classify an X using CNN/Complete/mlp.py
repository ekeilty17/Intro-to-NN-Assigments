import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_size, num_hidden_layers=1, hidden_size=25):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers 
        self.hidden_size = hidden_size if type(hidden_size) == list else [int(hidden_size)] * num_hidden_layers

        # Creating Hidden of neural network
        self.Hidden = nn.ModuleList()
        if self.num_hidden_layers == 0 or self.num_hidden_layers == []:
            self.last = nn.Sequential(
                nn.Linear(self.input_size, 1)
            )
        else:
            self.Hidden.append( 
                nn.Sequential(
                    nn.Linear(self.input_size, self.hidden_size[0]),
                    nn.ReLU()
                )
            )
            for l1, l2 in zip(self.hidden_size[:-1], self.hidden_size[1:]):
                self.Hidden.append( 
                        nn.Sequential(
                            nn.Linear(l1, l2),
                            nn.ReLU()
                        )
                    )
            self.last = nn.Sequential(
                nn.Linear(self.hidden_size[-1], 1)
            )

    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        for fc in self.Hidden:
            x = fc(x)
        return self.last(x)