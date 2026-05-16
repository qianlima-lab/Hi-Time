import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.dropout(x) + residual
        residual = x
        x = self.norm2(x)
        x = self.ff(x) + residual
        return x


class CrossVariableFusion(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.var_proj = nn.Linear(d_model, d_model)
        self.var_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, V, P, D = x.shape
        var_repr = x.mean(dim=2)
        var_proj = self.var_proj(var_repr)
        correlation = torch.bmm(var_proj, var_proj.transpose(1, 2))
        correlation = F.softmax(correlation / math.sqrt(D), dim=-1)
        x_reshaped = rearrange(x, 'b v p d -> (b p) v d')
        attn_out, _ = self.var_attn(x_reshaped, x_reshaped, x_reshaped)
        attn_out = self.dropout(attn_out)
        x_reshaped = self.norm(x_reshaped + attn_out)
        x = rearrange(x_reshaped, '(b p) v d -> b v p d', b=B, p=P)
        return x


class GatedPatchFusion(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, V, P, D = x.shape
        gate_weights = F.softmax(self.gate(x), dim=2)
        weighted = x * gate_weights
        fused = weighted.sum(dim=2).mean(dim=1)
        fused = self.norm(self.proj(fused))
        return fused
