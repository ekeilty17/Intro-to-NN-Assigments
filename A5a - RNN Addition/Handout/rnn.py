import torch
import torch.nn as nn

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.rnn_cell = nn.RNNCell(input_size=..., hidden_size=..., nonlinearity='relu')
        self.fc = nn.Linear(...)

    def init_hidden(self, batch_size=1):
        # need to return a tensor of size torch.Size[batch_size, self.hidden_size]
        raise NotImplementedError

    def forward(self, seqs):
        raise NotImplementedError