import os

import numpy as np

import torch
from torch.utils.data import Dataset


class BinarizedMNIST(Dataset):

    def __init__(self, data_dir, split='train', test=False):
        filename = os.path.join(data_dir, 'binarized_mnist_{}.amat'.format(split))

        if test:
            self.data = np.loadtxt(filename, max_rows=200)
        else:
            self.data = np.loadtxt(filename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.tensor(self.data[item]).view(-1, 28, 28).float()
