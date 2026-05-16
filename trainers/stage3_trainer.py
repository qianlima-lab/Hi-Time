import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict

from .base_trainer import BaseTrainer
from configs import Config
from models.encoder import Stage1Encoder
from models.discretizer import Stage2Discretizer
from models.predictor import Stage3GPT2Predictor


def mixup_code_embeds(code_embeds, codes, labels, alpha=0.4):
    """Apply mixup in code embedding space.
    
    Returns mixed code_embeds, mixed codes (as soft targets), labels_a, labels_b, lam.
    """
    if alpha <= 0:
        return code_embeds, codes, labels, labels, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    B = code_embeds[0].size(0)
    index = torch.randperm(B, device=code_embeds[0].device)
    
    mixed_embeds = [lam * ce + (1 - lam) * ce[index] for ce in code_embeds]
    # Codes can't be truly mixed (they're discrete), pass originals for code prediction loss
    return mixed_embeds, codes, labels, labels[index], lam


class Stage3Trainer(BaseTrainer):
    def __init__(self, config: Config, encoder: Stage1Encoder, discretizer: Stage2Discretizer,
                 device: torch.device):
        super().__init__(config, device, patience=config.training.stage3_patience)
        self.encoder = encoder.eval()
        self.discretizer = discretizer.eval()
        
        # Mixup config for Stage 3
        self.stage3_mixup_alpha = getattr(config.training, 'stage3_mixup_alpha', 0.0)
        
        # Create GPT-2 predictor with pretrained weights option
        self.model = Stage3GPT2Predictor(
            d_model=config.model.d_model,
            codebook_size=config.model.codebook_size,
            num_rq_layers=config.model.num_quantize_layers,
            num_classes=config.data.num_classes,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers,
            dropout=getattr(config.model, 'stage3_dropout', config.model.dropout),
            use_pretrained=config.model.use_pretrained_gpt2,
            pretrained_model=config.model.gpt2_model,
            freeze_gpt2=config.model.freeze_gpt2,
            unfreeze_layers=config.model.unfreeze_layers,
            bypass_gpt2_cls=getattr(config.model, 'bypass_gpt2_cls', False)
        ).to(device)
        
        # Print model parameter info
        param_info = self.model.get_num_parameters()
        print(f"\n[Stage 3] GPT-2 Predictor initialized:")
        print(f"  - Use pretrained: {config.model.use_pretrained_gpt2}")
        print(f"  - Pretrained model: {config.model.gpt2_model}")
        print(f"  - Total parameters: {param_info['total']:,}")
        print(f"  - GPT-2 parameters: {param_info['gpt2']:,}")
        print(f"  - Other parameters: {param_info['other']:,}")
        
        self.cls_weight = config.training.loss_weight_label
        self.code_weights = [
            config.training.loss_weight_code1,
            config.training.loss_weight_code2,
            config.training.loss_weight_code3,
        ][:config.model.num_quantize_layers]
        
        # Only optimize trainable parameters (frozen GPT-2 layers excluded)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.training.stage3_lr,
            weight_decay=config.training.stage3_weight_decay
        )
        self.scheduler = None
        stage3_use_sched = getattr(config.training, 'stage3_use_scheduler', config.training.use_scheduler)
        if stage3_use_sched:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.training.stage3_epochs,
                eta_min=config.training.scheduler_eta_min
            )
    
    def _get_codes_and_embeds(self, x: torch.Tensor):
        with torch.no_grad():
            repr = self.encoder.encode(x)
            disc_out = self.discretizer(repr)
        return disc_out['codes'], disc_out['code_embeds']
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            codes, code_embeds = self._get_codes_and_embeds(x)
            
            # Apply mixup in code embedding space if enabled
            if self.stage3_mixup_alpha > 0:
                code_embeds, codes, y_a, y_b, lam = mixup_code_embeds(
                    code_embeds, codes, y, self.stage3_mixup_alpha)
            else:
                y_a, y_b, lam = y, y, 1.0
            
            self.optimizer.zero_grad()
            outputs = self.model(code_embeds, y_a)
            
            if lam < 1.0:
                # Mixup loss: weighted combination of losses for both label sets
                loss_a, _ = self.model.get_loss(outputs, codes, y_a,
                                                cls_weight=self.cls_weight,
                                                code_weights=self.code_weights)
                loss_b, _ = self.model.get_loss(outputs, codes, y_b,
                                                cls_weight=self.cls_weight,
                                                code_weights=self.code_weights)
                loss = lam * loss_a + (1 - lam) * loss_b
            else:
                loss, _ = self.model.get_loss(outputs, codes, y,
                                              cls_weight=self.cls_weight,
                                              code_weights=self.code_weights)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_correct += (outputs['cls_logits'].argmax(dim=-1) == y_a).sum().item()
            total_samples += x.size(0)
        
        return {'loss': total_loss / total_samples, 'accuracy': total_correct / total_samples}
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        
        for x, y in val_loader:
            x, y = x.to(self.device), y.to(self.device)
            codes, code_embeds = self._get_codes_and_embeds(x)
            # No labels during validation — cls branch uses learnable CLS token
            outputs = self.model(code_embeds, labels=None)
            cls_loss = F.cross_entropy(outputs['cls_logits'], y)
            
            total_loss += cls_loss.item() * x.size(0)
            total_correct += (outputs['cls_logits'].argmax(dim=-1) == y).sum().item()
            total_samples += x.size(0)
        
        return {'loss': total_loss / total_samples, 'accuracy': total_correct / total_samples}


def train_stage3(config: Config, encoder: Stage1Encoder, discretizer: Stage2Discretizer,
                 train_loader: DataLoader, val_loader: DataLoader = None,
                 device: torch.device = None) -> Stage3Trainer:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Stage3Trainer(config, encoder, discretizer, device)
    trainer.train(train_loader, val_loader, config.training.stage3_epochs)
    return trainer
