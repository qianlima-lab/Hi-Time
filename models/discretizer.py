import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from .components.quantizers import ResidualQuantizer


class Stage2Discretizer(nn.Module):
    def __init__(self, d_model: int, codebook_size: int = 512, num_rq_layers: int = 3,
                 num_classes: int = 10, ema_decay: float = 0.99):
        super().__init__()
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.num_rq_layers = num_rq_layers
        self.num_classes = num_classes
        self.residual_quantizer = ResidualQuantizer(d_model, codebook_size, num_rq_layers, ema_decay)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        rq_output = self.residual_quantizer(repr)
        logits = self.classifier(rq_output['quantized'])
        return {
            'codes': rq_output['codes'],
            'quantized': rq_output['quantized'],
            'residuals': rq_output['residuals'],
            'code_embeds': rq_output['code_embeds'],
            'logits': logits
        }
    
    def get_loss(self, outputs: Dict, repr: torch.Tensor, labels: torch.Tensor,
                 commitment_weight: float = 0.25) -> tuple:
        cls_loss = F.cross_entropy(outputs['logits'], labels)
        commitment_loss = sum(
            F.mse_loss(res, torch.zeros_like(res)) for res in outputs['residuals']
        ) / len(outputs['residuals'])
        total_loss = cls_loss + commitment_weight * commitment_loss
        return total_loss, {
            'cls_loss': cls_loss.item(),
            'commitment_loss': commitment_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def get_code_embeddings(self, codes: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.residual_quantizer.get_code_embeddings(codes)
