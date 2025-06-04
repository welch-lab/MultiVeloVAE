import torch
from torch.utils.data import Dataset


class SCData(Dataset):
    def __init__(self, D, device=None):
        self.N, self.G = D.shape[0], D.shape[1]//3
        self.data = torch.tensor(D, dtype=torch.float, device=device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx


class SCDataE(Dataset):
    def __init__(self, D, E, device=None):
        self.N, self.G = D.shape[0], D.shape[1]//3
        self.data = torch.tensor(D, dtype=torch.float, device=device)
        self.data_e = torch.tensor(E, dtype=torch.float, device=device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx, self.data_e[idx]
