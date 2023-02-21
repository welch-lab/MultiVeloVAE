import numpy as np
from torch.utils.data import Dataset


class SCData(Dataset):
    def __init__(self, D, labels, c0=None, u0=None, s0=None, t0=None, weight=None, batch=None):
        self.N, self.G = D.shape[0], D.shape[1]//3
        self.data = D
        self.labels = labels
        self.batch = batch
        self.c0 = c0
        self.u0 = u0
        self.s0 = s0
        self.t0 = t0
        self.weight = np.ones((self.N, self.G)) if weight is None else weight

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.c0 is not None and self.u0 is not None and self.s0 is not None and self.t0 is not None:
            if self.batch is not None:
                return self.data[idx], self.labels[idx], self.weight[idx], idx, self.c0[idx], self.u0[idx], self.s0[idx], self.t0[idx], self.batch[idx]
            else:
                return self.data[idx], self.labels[idx], self.weight[idx], idx, self.c0[idx], self.u0[idx], self.s0[idx], self.t0[idx]
        if self.batch is not None:
            return self.data[idx], self.labels[idx], self.weight[idx], idx, self.batch[idx]
        else:
            return self.data[idx], self.labels[idx], self.weight[idx], idx


class SCTimedData(Dataset):
    def __init__(self, D, labels, t, c0=None, u0=None, s0=None, t0=None, weight=None, batch=None):
        self.N, self.G = D.shape[0], D.shape[1]//3
        self.data = D
        self.labels = labels
        self.batch = batch
        self.time = t.reshape(-1, 1)
        self.c0 = c0
        self.u0 = u0
        self.s0 = s0
        self.t0 = t0
        self.weight = np.ones((self.N, self.G)) if weight is None else weight

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.c0 is not None and self.u0 is not None and self.s0 is not None and self.t0 is not None:
            if self.batch is not None:
                return self.data[idx], self.labels[idx], self.time[idx], self.weight[idx], idx, self.c0[idx], self.u0[idx], self.s0[idx], self.t0[idx], self.batch[idx]
            else:
                return self.data[idx], self.labels[idx], self.time[idx], self.weight[idx], idx, self.c0[idx], self.u0[idx], self.s0[idx], self.t0[idx]
        if self.batch is not None:
            return self.data[idx], self.labels[idx], self.time[idx], self.weight[idx], idx, self.batch[idx]
        else:
            return self.data[idx], self.labels[idx], self.time[idx], self.weight[idx], idx
