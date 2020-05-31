import torch
import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class AdditionDataLoader(object):
    
    def __init__(self, N, seq_len, high, batch_size=None, drop_last=False, onehot=True):
        self.N = N
        self.seq_len = seq_len
        self.batch_size = N if batch_size is None else batch_size
        self.drop_last = drop_last

        # generating data
        self.data = np.random.randint(0, high, (self.N, self.seq_len))
        self.labels = np.sum(self.data, axis=1)

        # convert to pytorch tensor
        self.data = torch.tensor( self.data )
        self.labels = torch.tensor( self.labels )


        # one-hot encode data and labels
        if onehot:
            onehot_data = torch.zeros((self.data.size(0), seq_len, high), dtype=int)
            for i, seq in enumerate(self.data):
                onehot_data[i] = self.onehot_encode(seq, seq_len, high)
            self.data = onehot_data
        else:
            self.data = self.data.unsqueeze(2)

        # initializing counting parameters
        self.index = 0

    @staticmethod
    def onehot_encode(seq, seq_len, max_val):
        onehot_seq = torch.zeros((seq_len, max_val), dtype=int)
        for onehot, n in zip(onehot_seq, seq):
            onehot[n] = 1
        return onehot_seq

    def __iter__(self):
        return self
    
    def __next__(self):
        
        # stop condition
        if self.index >= self.N:
            self.index = 0          # resetting index for next iteration
            raise StopIteration

        # iterating
        self.index += self.batch_size
    
        if self.index > self.N:
            if self.drop_last:
                self.index = 0      # resetting index for next iteration
                raise StopIteration
            else:
                return self.data[self.index - self.batch_size: ], self.labels[self.index - self.batch_size: ]
        else:
            return self.data[self.index - self.batch_size: self.index], self.labels[self.index - self.batch_size: self.index]


def load_data(seq_len, high=100, batch_size=None, onehot=True):
    train_loader = AdditionDataLoader(9000, seq_len, high=high, batch_size=batch_size, onehot=onehot)
    valid_loader = AdditionDataLoader(1000, seq_len, high=high, batch_size=None, onehot=onehot)

    return train_loader, valid_loader

def get_n_samples(loader, n):
    for data, labels in loader:
        return data[0:n], labels[0:n]