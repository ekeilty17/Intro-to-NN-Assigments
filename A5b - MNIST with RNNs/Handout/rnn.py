import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # tanh is the default nonlinearity, but you can experiment with relu if you would like
        self.rnn = nn.RNN(  input_size=..., hidden_size=..., \
                            num_layers=..., nonlinearity='tanh'  )
        
        self.fc = nn.Linear(..., ...)
        
    def init_hidden(self, batch_size=1):
        # need to return a tensor of size torch.Size[num_layers, batch_size, hidden_size]
        raise NotImplementedError
        
    def forward(self, x):
        raise NotImplementedError