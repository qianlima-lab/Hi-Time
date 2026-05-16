import torch
import torch.nn as nn
import math
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, seq_len: int, device: torch.device = None) -> torch.Tensor:
        if device is not None and self.pe.device != device:
            return self.pe[:seq_len].to(device)
        return self.pe[:seq_len]


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int, stride: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, V, T = x.shape
        num_patches = (T - self.patch_len) // self.stride + 1
        patches = []
        for i in range(num_patches):
            start = i * self.stride
            patches.append(x[:, :, start:start + self.patch_len])
        patches = torch.stack(patches, dim=2)
        patches = self.dropout(self.norm(self.proj(patches)))
        return patches
    
    def get_num_patches(self, seq_len: int) -> int:
        return (seq_len - self.patch_len) // self.stride + 1


class TimeSeriesPatchEmbedding(nn.Module):
    def __init__(self, n_vars: int, seq_len: int, patch_len: int, stride: int, gpt2_hidden: int = 768):
        super().__init__()
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.proj = nn.Linear(n_vars * patch_len, gpt2_hidden)
        self.norm = nn.LayerNorm(gpt2_hidden)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, gpt2_hidden) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, V, T = x.shape
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            patch = x[:, :, start:start + self.patch_len]
            patch = rearrange(patch, 'b v p -> b (v p)')
            patches.append(patch)
        patches = torch.stack(patches, dim=1)
        patches = self.norm(self.proj(patches)) + self.pos_embed
        return patches
