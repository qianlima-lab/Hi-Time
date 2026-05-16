import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Dict

from .components.embeddings import PatchEmbedding, PositionalEncoding
from .components.attention import TransformerBlock, CrossVariableFusion, GatedPatchFusion


class ScalePatchTST(nn.Module):
    def __init__(self, patch_len: int, stride: int, d_model: int, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_len, stride, d_model, dropout)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, V, T = x.shape
        patches = self.patch_embed(x)
        num_patches = patches.size(2)
        pos_embed = self.pos_encoding(num_patches, patches.device)
        patches = patches + pos_embed.unsqueeze(0).unsqueeze(0)
        patches = rearrange(patches, 'b v p d -> (b v) p d')
        for layer in self.layers:
            patches = layer(patches)
        patches = rearrange(patches, '(b v) p d -> b v p d', b=B, v=V)
        return patches


class MultiScaleEncoder(nn.Module):
    def __init__(self, scales: List[int], patch_len: int, stride: int, d_model: int,
                 n_heads: int = 4, n_layers: int = 2, d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.scales = scales
        self.d_model = d_model
        self.scale_encoders = nn.ModuleList()
        self.scale_mlps = nn.ModuleList()
        for scale in scales:
            scaled_patch_len = max(patch_len // scale, 4)
            scaled_stride = max(stride // scale, 2)
            self.scale_encoders.append(ScalePatchTST(
                scaled_patch_len, scaled_stride, d_model, n_heads, n_layers, d_ff, dropout
            ))
            self.scale_mlps.append(nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
            ))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, V, T = x.shape
        scale_features = []
        min_patches = float('inf')
        for i, scale in enumerate(self.scales):
            x_scaled = F.avg_pool1d(x, kernel_size=scale, stride=scale) if scale > 1 else x
            features = self.scale_mlps[i](self.scale_encoders[i](x_scaled))
            scale_features.append(features)
            min_patches = min(min_patches, features.size(2))
        aligned_features = []
        for feat in scale_features:
            if feat.size(2) > min_patches:
                feat = rearrange(feat, 'b v p d -> (b v) d p')
                feat = F.adaptive_avg_pool1d(feat, min_patches)
                feat = rearrange(feat, '(b v) d p -> b v p d', b=B, v=V)
            aligned_features.append(feat)
        return sum(aligned_features) / len(aligned_features)


class Stage1Encoder(nn.Module):
    def __init__(self, seq_len: int, n_vars: int, num_classes: int, scales: List[int] = None,
                 patch_len: int = 16, stride: int = 8, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        if scales is None:
            scales = [1, 2, 4]
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.d_model = d_model
        self.num_classes = num_classes
        self.multi_scale = MultiScaleEncoder(scales, patch_len, stride, d_model, n_heads, n_layers, d_ff, dropout)
        self.cross_var = CrossVariableFusion(d_model, n_heads, dropout)
        self.gated_fusion = GatedPatchFusion(d_model, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale_feat = self.multi_scale(x)
        cross_var_feat = self.cross_var(multi_scale_feat)
        return self.gated_fusion(cross_var_feat)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        repr = self.encode(x)
        return {'logits': self.classifier(repr), 'repr': repr}
    
    def get_loss(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> tuple:
        loss = F.cross_entropy(outputs['logits'], labels)
        return loss, {'cls_loss': loss.item()}
