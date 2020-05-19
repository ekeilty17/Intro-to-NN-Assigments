import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import struct
import os
import gzip
from array import array


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class MNISTDataset(Dataset):

    train_data_fname = "train-images-idx3-ubyte.gz"
    train_labels_fname = "train-labels-idx1-ubyte.gz"

    valid_data_fname = "t10k-images-idx3-ubyte.gz"
    valid_labels_fname = "t10k-labels-idx1-ubyte.gz"

    def __init__(self, root, train_dataset=True):
        self.root = root
        
        self.train_data_path = os.path.join(self.root, self.train_data_fname)
        self.train_labels_path = os.path.join(self.root, self.train_labels_fname)

        self.valid_data_path = os.path.join(self.root, self.valid_data_fname)
        self.valid_labels_path = os.path.join(self.root, self.valid_labels_fname)

        if train_dataset:
            self.data, self.labels = self.load_from_path(self.train_data_path, self.train_labels_path)
        else:
            self.data, self.labels = self.load_from_path(self.valid_data_path, self.valid_labels_path)

        # we need to reshape the data to add a 1 to represent the channels for nn.Conv2d
        self.data = self.data.unsqueeze(1)


    """ How this works, because the code is not very readable
    1)  Gets path to ubyte file
    2)  upzip the .gz file at that path. f_img/f_lbl are of type '_io.BufferedReader'
    3)  The file that we opened is a ubyte file, so we need something to correctly interpret it
        This is what struct.unpack() does
            - f_img.read(16) reads the first 16 bits of the file. Likewise f_lbl.read(8) reads the first 8 bits.
              The first few bytes of the file is not data, it is information about the file
            - '>IIII' and '>II' tells struct.unpack how to unpack the data. So the 'IIII' means give a tuple with 4 items in it 
            and the '>' gives the byte order. '>' = big-endian, '<' = little-endian (google it)
            - The magic number is a unique number at the beginning of a file which we could use to verify the file we asked for is the file we got from the OS
            - N = number of samples
            - rows, cols = dimensions of image
    4) 'B' = unsigned char, 'b' = signed char. It's just the datatypes of the files. f_img.read() and f_lbl.read() reads the entire file
    5) convert images/labels to tensors and reshape if necessary
    """
    @staticmethod
    def load_from_path(data_path, label_path):
        
        with gzip.open(data_path, 'rb') as f_img:
            magic_num, N, rows, cols = struct.unpack(">IIII", f_img.read(16))
            img = array("B", f_img.read())
        images = torch.tensor(img).reshape(N, rows, cols)

        with gzip.open(label_path, 'rb') as f_lbl:
            magic_num, N = struct.unpack(">II", f_lbl.read(8))
            lbl = array("b", f_lbl.read())
        labels = torch.tensor(lbl)

        return images, labels

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index, :], self.labels[index]


def load_data(batch_size=None, seed=None):
    # for when the data gets shuffled
    if not seed is None:
        torch.manual_seed(seed)

    train_dataset = MNISTDataset("../data", train_dataset=True)
    batch_size = len(train_dataset) if batch_size is None else batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = MNISTDataset("../data", train_dataset=False)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))

    return train_loader, valid_loader