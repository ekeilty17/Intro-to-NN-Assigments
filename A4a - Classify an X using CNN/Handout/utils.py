import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class XPatternDataset(Dataset):

    def __init__(self, data, labels):
        n = int(np.sqrt(data.shape[1]))
        data = data.reshape(-1, 1, n, n)        # we need the extra 1 for nn.Conv2d to work correctly
        
        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index, :], self.labels[index]


def load_data(batch_size=None, seed=None):
    train_data = np.loadtxt(open("../data/traindata.csv", "r"), delimiter=",")
    train_labels = np.loadtxt(open("../data/trainlabels.csv", "r"), delimiter=",")
    valid_data = np.loadtxt(open("../data/validdata.csv", "r"), delimiter=",")
    valid_labels = np.loadtxt(open("../data/validlabels.csv", "r"), delimiter=",")

    train_dataset = XPatternDataset(train_data, train_labels)
    batch_size = len(train_dataset) if batch_size is None else batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = XPatternDataset(valid_data, valid_labels)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))

    return train_loader, valid_loader

def get_n_samples(loader, n=1):
    for data, labels in loader:
        return data[0:n], labels[0:n]