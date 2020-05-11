import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_data():
    train_data = np.loadtxt(open("../data/traindata.csv", "rb"), delimiter=",")
    train_labels = np.loadtxt(open("../data/trainlabels.csv", "rb"), delimiter=",")
    valid_data = np.loadtxt(open("../data/validdata.csv", "rb"), delimiter=",")
    valid_labels = np.loadtxt(open("../data/validlabels.csv", "rb"), delimiter=",")

    return train_data, train_labels, valid_data, valid_labels
