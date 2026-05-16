from dataclasses import dataclass, field
from typing import List, Dict
import json
from pathlib import Path


@dataclass
class DataConfig:
    dataset_name: str = 'FingerMovements'
    data_dir: str = './data/datasets'
    normalize: bool = True
    batch_size: int = 32
    val_ratio: float = 0.2
    num_workers: int = 0
    seq_len: int = 50
    n_vars: int = 28
    num_classes: int = 2


@dataclass
class ModelConfig:
    d_model: int = 128
    dropout: float = 0.1
    stage3_dropout: float = 0.2  # Separate dropout for Stage 3 GPT-2 predictor
    scales: List[int] = field(default_factory=lambda: [1, 2, 4])
    patch_len: int = 8
    stride: int = 2
    n_heads: int = 4
    n_layers: int = 2
    d_ff_multiplier: int = 2
    codebook_size: int = 64
    num_quantize_layers: int = 3
    ema_decay: float = 0.99
    gpt2_model: str = './pretrained/gpt2'
    use_pretrained_gpt2: bool = True
    freeze_gpt2: bool = False
    unfreeze_layers: int = 12
    deep_classifier: bool = True
    bypass_gpt2_cls: bool = False  # If True, classifier uses only code_embeds, not GPT-2 output
    
    @property
    def d_ff(self) -> int:
        return self.d_model * self.d_ff_multiplier


@dataclass
class TrainingConfig:
    seed: int = 10
    grad_clip: float = 1.0
    scheduler_eta_min: float = 1e-6
    use_scheduler: bool = True  # legacy global flag (prefer per-stage flags below)
    stage1_use_scheduler: bool = True
    stage2_use_scheduler: bool = True
    stage3_use_scheduler: bool = False
    stage1_epochs: int = 250
    stage1_lr: float = 1e-3
    stage1_weight_decay: float = 1e-4
    stage1_patience: int = 250  # effectively disabled for stage1; always run full epochs
    mixup_alpha: float = 0.4
    label_smoothing: float = 0.1
    stage2_epochs: int = 50
    stage2_lr: float = 1e-3
    stage2_weight_decay: float = 1e-4
    stage3_epochs: int = 200
    stage3_lr: float = 5e-6
    stage3_weight_decay: float = 1e-4
    stage3_patience: int = 100
    num_runs: int = 1
    merge_resplit: bool = True  # paper: 60/20/20 resplit
    loss_weight_code1: float = 0.1
    loss_weight_code2: float = 0.1
    loss_weight_code3: float = 0.1
    loss_weight_label: float = 2.0
    stage3_mixup_alpha: float = 0.0  # Mixup alpha for Stage 3 code embeddings (0 = disabled)
    
    @property
    def loss_weights(self) -> Dict[str, float]:
        return {
            'code1': self.loss_weight_code1,
            'code2': self.loss_weight_code2,
            'code3': self.loss_weight_code3,
            'label': self.loss_weight_label
        }


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint_dir: str = './checkpoints'
    
    @property
    def seed(self) -> int:
        return self.training.seed
    
    @property
    def dataset_name(self) -> str:
        return self.data.dataset_name
    
    @property
    def data_dir(self) -> str:
        return self.data.data_dir
    
    @property
    def normalize(self) -> bool:
        return self.data.normalize
    
    @property
    def batch_size(self) -> int:
        return self.data.batch_size
    
    @property
    def val_ratio(self) -> float:
        return self.data.val_ratio
    
    @property
    def num_workers(self) -> int:
        return self.data.num_workers
    
    @property
    def d_model(self) -> int:
        return self.model.d_model
    
    @property
    def dropout(self) -> float:
        return self.model.dropout
    
    @property
    def scales(self) -> List[int]:
        return self.model.scales
    
    @property
    def patch_len(self) -> int:
        return self.model.patch_len
    
    @property
    def stride(self) -> int:
        return self.model.stride
    
    @property
    def n_heads(self) -> int:
        return self.model.n_heads
    
    @property
    def n_layers(self) -> int:
        return self.model.n_layers
    
    @property
    def d_ff(self) -> int:
        return self.model.d_ff
    
    @property
    def codebook_size(self) -> int:
        return self.model.codebook_size
    
    @property
    def num_quantize_layers(self) -> int:
        return self.model.num_quantize_layers
    
    @property
    def gpt2_model(self) -> str:
        return self.model.gpt2_model
    
    @property
    def use_pretrained_gpt2(self) -> bool:
        return self.model.use_pretrained_gpt2
    
    @property
    def freeze_gpt2(self) -> bool:
        return self.model.freeze_gpt2
    
    @property
    def unfreeze_layers(self) -> int:
        return self.model.unfreeze_layers
    
    @property
    def stage1_epochs(self) -> int:
        return self.training.stage1_epochs
    
    @property
    def stage1_lr(self) -> float:
        return self.training.stage1_lr
    
    @property
    def stage1_weight_decay(self) -> float:
        return self.training.stage1_weight_decay
    
    @property
    def stage1_patience(self) -> int:
        return self.training.stage1_patience
    
    @property
    def stage2_epochs(self) -> int:
        return self.training.stage2_epochs
    
    @property
    def stage2_lr(self) -> float:
        return self.training.stage2_lr
    
    @property
    def stage2_weight_decay(self) -> float:
        return self.training.stage2_weight_decay
    
    @property
    def stage3_epochs(self) -> int:
        return self.training.stage3_epochs
    
    @property
    def stage3_lr(self) -> float:
        return self.training.stage3_lr
    
    @property
    def stage3_weight_decay(self) -> float:
        return self.training.stage3_weight_decay
    
    @property
    def stage3_patience(self) -> int:
        return self.training.stage3_patience
    
    @property
    def grad_clip(self) -> float:
        return self.training.grad_clip
    
    @property
    def scheduler_eta_min(self) -> float:
        return self.training.scheduler_eta_min
    
    @property
    def loss_weights(self) -> Dict[str, float]:
        return self.training.loss_weights
    
    def print_config(self):
        print("\n" + "=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(f"Seed: {self.seed}")
        print(f"\n[Dataset]")
        print(f"  Name: {self.dataset_name}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Val Ratio: {self.val_ratio}")
        print(f"\n[Model]")
        print(f"  d_model: {self.d_model}")
        print(f"  Scales: {self.scales}")
        print(f"  Patch: len={self.patch_len}, stride={self.stride}")
        print(f"  Transformer: heads={self.n_heads}, layers={self.n_layers}")
        print(f"  Scheduler: {self.training.use_scheduler}")
        print(f"  Runs: {self.training.num_runs}, Resplit: {self.training.merge_resplit}")
        print(f"\n[Stage 1]")
        print(f"  Epochs: {self.stage1_epochs}, LR: {self.stage1_lr}, WD: {self.stage1_weight_decay}")
        print(f"\n[Stage 2]")
        print(f"  Codebook Size: {self.codebook_size}")
        print(f"  Quantize Layers: {self.num_quantize_layers}")
        print(f"  Epochs: {self.stage2_epochs}, LR: {self.stage2_lr}, WD: {self.stage2_weight_decay}")
        print(f"\n[Stage 3]")
        print(f"  GPT-2 Model: {self.gpt2_model}")
        print(f"  Freeze GPT-2: {self.freeze_gpt2}, Unfreeze Layers: {self.unfreeze_layers}")
        print(f"  Epochs: {self.stage3_epochs}, LR: {self.stage3_lr}, WD: {self.stage3_weight_decay}")
        print(f"  Loss Weights: {self.loss_weights}")
        print("=" * 60 + "\n")
    
    def to_dict(self) -> Dict:
        return {
            'data': {
                'dataset_name': self.data.dataset_name,
                'data_dir': self.data.data_dir,
                'normalize': self.data.normalize,
                'batch_size': self.data.batch_size,
                'val_ratio': self.data.val_ratio,
                'num_workers': self.data.num_workers,
            },
            'model': {
                'd_model': self.model.d_model,
                'dropout': self.model.dropout,
                'scales': list(self.model.scales),
                'patch_len': self.model.patch_len,
                'stride': self.model.stride,
                'n_heads': self.model.n_heads,
                'n_layers': self.model.n_layers,
                'd_ff_multiplier': self.model.d_ff_multiplier,
                'codebook_size': self.model.codebook_size,
                'num_quantize_layers': self.model.num_quantize_layers,
                'ema_decay': self.model.ema_decay,
                'gpt2_model': self.model.gpt2_model,
                'freeze_gpt2': self.model.freeze_gpt2,
                'unfreeze_layers': self.model.unfreeze_layers,
            },
            'training': {
                'seed': self.training.seed,
                'grad_clip': self.training.grad_clip,
                'scheduler_eta_min': self.training.scheduler_eta_min,
                'use_scheduler': self.training.use_scheduler,
                'stage1_epochs': self.training.stage1_epochs,
                'stage1_lr': self.training.stage1_lr,
                'stage1_weight_decay': self.training.stage1_weight_decay,
                'stage1_patience': self.training.stage1_patience,
                'stage2_epochs': self.training.stage2_epochs,
                'stage2_lr': self.training.stage2_lr,
                'stage2_weight_decay': self.training.stage2_weight_decay,
                'stage3_epochs': self.training.stage3_epochs,
                'stage3_lr': self.training.stage3_lr,
                'stage3_weight_decay': self.training.stage3_weight_decay,
                'stage3_patience': self.training.stage3_patience,
                'num_runs': self.training.num_runs,
                'merge_resplit': self.training.merge_resplit,
                'loss_weight_code1': self.training.loss_weight_code1,
                'loss_weight_code2': self.training.loss_weight_code2,
                'loss_weight_code3': self.training.loss_weight_code3,
                'loss_weight_label': self.training.loss_weight_label,
                'mixup_alpha': self.training.mixup_alpha,
                'label_smoothing': self.training.label_smoothing,
            },
            'checkpoint_dir': self.checkpoint_dir,
        }
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            checkpoint_dir=config_dict.get('checkpoint_dir', './checkpoints')
        )
