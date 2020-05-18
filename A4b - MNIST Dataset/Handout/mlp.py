import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()

        raise NotImplementedError

        # can be exactly the same as previous assignments 
        # or you can just make one custom architecture for this classification task

    def forward(self, x):
        # x.size() = torch.Size([opts.batch_size, 28, 28])
        raise NotImplementedError

        # Hint: you might have to reshape x
        # google torch.reshape()