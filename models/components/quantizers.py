import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict


class KMeansQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int, ema_decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.ema_decay = ema_decay
        self.eps = eps
        self.register_buffer('codebook', torch.randn(codebook_size, dim))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_avg', self.codebook.clone())
        self.register_buffer('initted', torch.tensor(False))
    
    def init_codebook(self, data: torch.Tensor):
        if self.initted:
            return
        n_samples = data.size(0)
        if n_samples >= self.codebook_size:
            indices = torch.randperm(n_samples)[:self.codebook_size]
            self.codebook.data.copy_(data[indices])
        else:
            repeats = (self.codebook_size // n_samples) + 1
            expanded = data.repeat(repeats, 1)[:self.codebook_size]
            self.codebook.data.copy_(expanded + torch.randn_like(expanded) * 0.01)
        self.embed_avg.data.copy_(self.codebook.data)
        self.initted.fill_(True)
    
    def forward(self, x: torch.Tensor, code_embedding: nn.Embedding = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.initted and self.training:
            self.init_codebook(x.detach())
            # Also init the code_embedding if provided
            if code_embedding is not None:
                with torch.no_grad():
                    code_embedding.weight.data.copy_(self.codebook.data)
        dist = torch.cdist(x, self.codebook)
        indices = dist.argmin(dim=-1)
        quantized = F.embedding(indices, self.codebook)
        if self.training:
            self._ema_update(x, indices, code_embedding)
        residual = x - quantized
        quantized = x + (quantized - x).detach()
        return quantized, indices, residual
    
    def _ema_update(self, x: torch.Tensor, indices: torch.Tensor, code_embedding: nn.Embedding = None):
        with torch.no_grad():
            one_hot = F.one_hot(indices, self.codebook_size).float()
            cluster_size = one_hot.sum(dim=0)
            self.cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
            embed_sum = one_hot.T @ x
            self.embed_avg.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)
            n = self.cluster_size.clamp(min=self.eps)
            self.codebook.data.copy_(self.embed_avg / n.unsqueeze(-1))
            
            # Revive dead codes: replace codes with cluster_size < threshold
            dead_mask = self.cluster_size < 0.5
            n_dead = dead_mask.sum().item()
            if n_dead > 0 and x.size(0) > 0:
                # Replace dead codes with random samples from current batch + noise
                rand_idx = torch.randint(0, x.size(0), (n_dead,), device=x.device)
                self.codebook.data[dead_mask] = x[rand_idx] + torch.randn(n_dead, self.dim, device=x.device) * 0.01
                self.embed_avg.data[dead_mask] = self.codebook.data[dead_mask]
                self.cluster_size[dead_mask] = 1.0
            
            # Sync code_embedding weights with codebook
            if code_embedding is not None:
                code_embedding.weight.data.copy_(self.codebook.data)


class ResidualQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int = 512, num_layers: int = 3, ema_decay: float = 0.99):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        self.quantizers = nn.ModuleList([
            KMeansQuantizer(dim, codebook_size, ema_decay) for _ in range(num_layers)
        ])
        self.code_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, dim) for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        codes, residuals, code_embeds = [], [], []
        residual = x
        quantized_sum = torch.zeros_like(x)
        for i, quantizer in enumerate(self.quantizers):
            quantized, indices, new_residual = quantizer(residual, self.code_embeddings[i])
            codes.append(indices)
            residuals.append(new_residual)  # store the actual residual after quantization
            code_embeds.append(self.code_embeddings[i](indices))
            quantized_sum = quantized_sum + quantized
            residual = new_residual
        return {'codes': codes, 'quantized': quantized_sum, 'residuals': residuals, 'code_embeds': code_embeds}
    
    def get_code_embeddings(self, codes: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.code_embeddings[i](code) for i, code in enumerate(codes)]
    
    def decode(self, codes: List[torch.Tensor]) -> torch.Tensor:
        quantized_sum = None
        for i, code in enumerate(codes):
            quantized = F.embedding(code, self.quantizers[i].codebook)
            quantized_sum = quantized if quantized_sum is None else quantized_sum + quantized
        return quantized_sum
