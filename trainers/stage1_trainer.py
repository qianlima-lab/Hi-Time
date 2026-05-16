import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict

from .base_trainer import BaseTrainer
from configs import Config
from models.encoder import Stage1Encoder


def mixup_data(x, y, alpha=0.4):
    """Apply mixup augmentation to a batch."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


class Stage1Trainer(BaseTrainer):
    def __init__(self, config: Config, device: torch.device):
        super().__init__(config, device, patience=config.training.stage1_patience)
        d_model = config.model.d_model
        dropout = config.model.dropout
        num_classes = config.data.num_classes
        self.model = Stage1Encoder(
            seq_len=config.data.seq_len,
            n_vars=config.data.n_vars,
            num_classes=num_classes,
            scales=config.model.scales,
            patch_len=config.model.patch_len,
            stride=config.model.stride,
            d_model=d_model,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers,
            d_ff=config.model.d_ff,
            dropout=dropout
        ).to(device)
        # Replace classifier with a deeper version for stronger regularization
        if config.model.deep_classifier:
            cls_dropout = min(dropout * 2, 0.3)
            self.model.classifier = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(cls_dropout),
                nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(cls_dropout),
                nn.Linear(d_model // 2, num_classes)
            ).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.stage1_lr,
            weight_decay=config.training.stage1_weight_decay
        )
        self.scheduler = None
        stage1_use_sched = getattr(config.training, 'stage1_use_scheduler', config.training.use_scheduler)
        if stage1_use_sched:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.training.stage1_epochs,
                eta_min=config.training.scheduler_eta_min
            )
        self.mixup_alpha = config.training.mixup_alpha
        self.label_smoothing = config.training.label_smoothing
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            # Apply mixup
            mixed_x, y_a, y_b, lam = mixup_data(x, y, self.mixup_alpha)
            
            self.optimizer.zero_grad()
            outputs = self.model(mixed_x)
            
            # Mixup loss with label smoothing
            loss = lam * F.cross_entropy(outputs['logits'], y_a, label_smoothing=self.label_smoothing) + \
                   (1 - lam) * F.cross_entropy(outputs['logits'], y_b, label_smoothing=self.label_smoothing)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            # For accuracy tracking, use unmixed predictions vs original labels
            total_correct += (outputs['logits'].argmax(dim=-1) == y_a).sum().item()
            total_samples += x.size(0)
        
        return {'loss': total_loss / total_samples, 'accuracy': total_correct / total_samples}
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        
        for x, y in val_loader:
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.model(x)
            loss, _ = self.model.get_loss(outputs, y)
            
            total_loss += loss.item() * x.size(0)
            total_correct += (outputs['logits'].argmax(dim=-1) == y).sum().item()
            total_samples += x.size(0)
        
        return {'loss': total_loss / total_samples, 'accuracy': total_correct / total_samples}


def train_stage1(config: Config, train_loader: DataLoader, val_loader: DataLoader = None,
                 device: torch.device = None) -> Stage1Trainer:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Stage1Trainer(config, device)
    trainer.train(train_loader, val_loader, config.training.stage1_epochs)
    return trainer
