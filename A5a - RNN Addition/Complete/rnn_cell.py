import torch
import torch.nn as nn

class RNNCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.hidden_size = hidden_size

        self.rnn_cell = nn.RNNCell(  input_size=input_size, hidden_size=hidden_size, nonlinearity='relu')

    def forward(self, x, hidden):
        return self.rnn_cell(x, hidden)

    def init_hidden(self, batch_size=1):
        return torch.ones(batch_size, self.hidden_size)