import torch
from torch.utils.data import Dataset


class SCData(Dataset):
    def __init__(self, D, labels, weight=None, batch=None, device=None):
        self.N, self.G = D.shape[0], D.shape[1]//3
        self.data = torch.tensor(D, dtype=torch.float, device=device)
        self.labels = labels
        self.batch = torch.tensor(batch, dtype=int, device=device) if batch is not None else None
        self.weight = torch.tensor(weight, dtype=torch.float, device=device) if weight is not None else torch.ones((self.N, self.G), dtype=torch.float, device=device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.batch is not None:
            return self.data[idx], self.labels[idx], self.weight[idx], idx, self.batch[idx]
        else:
            return self.data[idx], self.labels[idx], self.weight[idx], idx
