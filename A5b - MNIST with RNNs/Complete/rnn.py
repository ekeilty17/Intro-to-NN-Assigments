import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(  input_size=input_size, hidden_size=hidden_size, \
                            num_layers=num_layers, nonlinearity='tanh'  )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def init_hidden(self, batch_size=1):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
    def forward(self, x):
        batch_size = x.size(1)
        hidden = self.init_hidden(batch_size)
        output, hidden = self.rnn(x, hidden)      
        return self.fc(hidden).reshape(batch_size, -1)