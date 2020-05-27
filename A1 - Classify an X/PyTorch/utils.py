import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class XPatternDataset(Dataset):

    def __init__(self, data_path, label_path):
        self.data = torch.tensor( pd.read_csv(data_path, header=None).values )
        self.labels = torch.tensor( pd.read_csv(label_path, header=None).values ).squeeze()
    
    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, index):
        return self.data[index, :], self.labels[index]


def load_data(batch_size=None):
    train_dataset = XPatternDataset("../data/traindata.csv", "../data/trainlabels.csv")
    batch_size = len(train_dataset) if batch_size is None else batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = XPatternDataset("../data/validdata.csv", "../data/validlabels.csv")
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))

    return train_loader, valid_loader

def get_n_samples(loader, n=1):
    for data, labels in loader:
        return data[0:n], labels[0:n]