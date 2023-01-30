import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr_t = f['lr'][idx] / 65535.
            return lr_t, hr_t


    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr_e = f['lr'][str(idx)][:, :] / 65535.
            hr_e = f['hr'][str(idx)][:, :] / 65535.
            return lr_e, hr_e

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

