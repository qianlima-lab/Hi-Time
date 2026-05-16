import torch
import numpy as np
from typing import List, Callable, Union


def normalize_data(data: Union[np.ndarray, torch.Tensor], dim: int = -1) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(data, np.ndarray):
        mean = data.mean(axis=dim, keepdims=True)
        std = data.std(axis=dim, keepdims=True) + 1e-8
        return (data - mean) / std
    else:
        mean = data.mean(dim=dim, keepdim=True)
        std = data.std(dim=dim, keepdim=True) + 1e-8
        return (data - mean) / std


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class AddNoise:
    def __init__(self, noise_level: float = 0.1):
        self.noise_level = noise_level
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.noise_level * torch.randn_like(x)


class RandomScale:
    def __init__(self, scale_range: tuple = (0.9, 1.1)):
        self.scale_range = scale_range
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        return x * scale


class TimeWarp:
    def __init__(self, sigma: float = 0.2):
        self.sigma = sigma
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x
