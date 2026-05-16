import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Callable


class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, transform: Optional[Callable] = None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.labels[idx]
