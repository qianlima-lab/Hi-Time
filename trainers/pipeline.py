import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional

from configs import Config
from utils.helpers import set_seed
from .stage1_trainer import train_stage1
from .stage2_trainer import train_stage2
from .stage3_trainer import train_stage3


class ThreeStagePipeline:
    def __init__(self, config: Config, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stage1_trainer = None
        self.discretizer = None
        self.stage3_trainer = None
    
    def run(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict:
        seed = self.config.training.seed
        
        print("=" * 50)
        print("Stage 1: Training Multi-Scale Encoder")
        print("=" * 50)
        self.stage1_trainer = train_stage1(self.config, train_loader, val_loader, self.device)
        
        # Reset seed before each stage for reproducibility
        set_seed(seed)
        print("\n" + "=" * 50)
        print("Stage 2: Training RQ-KMeans Discretizer")
        print("=" * 50)
        self.discretizer = train_stage2(
            self.config, self.stage1_trainer.model, train_loader, val_loader, self.device
        )
        
        # Reset seed before Stage 3 for reproducibility
        set_seed(seed)
        print("\n" + "=" * 50)
        print("Stage 3: Training GPT-2 Predictor")
        print("=" * 50)
        self.stage3_trainer = train_stage3(
            self.config, self.stage1_trainer.model, self.discretizer,
            train_loader, val_loader, self.device
        )
        
        return {
            'encoder': self.stage1_trainer.model,
            'discretizer': self.discretizer,
            'predictor': self.stage3_trainer.model
        }
    
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        self.stage1_trainer.model.eval()
        self.discretizer.eval()
        self.stage3_trainer.model.eval()
        
        total_correct, total_samples = 0, 0
        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            repr = self.stage1_trainer.model.encode(x)
            disc_out = self.discretizer(repr)
            # Don't pass labels during evaluation
            outputs = self.stage3_trainer.model(disc_out['code_embeds'], labels=None)
            total_correct += (outputs['cls_logits'].argmax(dim=-1) == y).sum().item()
            total_samples += x.size(0)
        
        return {'accuracy': total_correct / total_samples}
