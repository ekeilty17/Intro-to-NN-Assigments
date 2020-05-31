import torch
import torch.nn as nn

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        # The standard non-linearity is tanh, but I am using relu as it matches the manual solution
        self.rnn_cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden_size)

    def forward(self, seqs):
        hidden = self.init_hidden(seqs.size(1))
        for x in seqs:
            hidden = self.rnn_cell(x.float(), hidden)
        return self.fc(hidden)