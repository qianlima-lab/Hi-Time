import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Dict, Optional
import os

from configs import Config
from utils.metrics import MetricTracker, EarlyStopping
from utils.helpers import save_checkpoint


class BaseTrainer(ABC):
    def __init__(self, config: Config, device: torch.device, patience: int = 20):
        self.config = config
        self.device = device
        self.metric_tracker = MetricTracker()
        self.early_stopping = EarlyStopping(patience=patience, min_delta=1e-4, mode='max')
        self.scheduler = None  # subclasses may set this
    
    @abstractmethod
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        pass
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              num_epochs: Optional[int] = None) -> Dict[str, list]:
        num_epochs = num_epochs or self.config.training.num_epochs
        history = {'train': [], 'val': []}
        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader)
            history['train'].append(train_metrics)
            self.metric_tracker.update(train_metrics, prefix='train')
            
            # Step scheduler after each epoch
            if self.scheduler is not None:
                self.scheduler.step()
            
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val'].append(val_metrics)
                self.metric_tracker.update(val_metrics, prefix='val')
                
                val_acc = val_metrics.get('accuracy', 0)
                if self.early_stopping(val_acc, model=self.model):
                    print(f"Early stopping at epoch {epoch + 1}")
                    self.early_stopping.restore_best(self.model)
                    break
            
            log_interval = max(1, num_epochs // 10)  # ~10 log lines per stage
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                self._log_progress(epoch + 1, train_metrics, val_metrics if val_loader else None)
        else:
            # Training finished without early stopping — restore best model
            if val_loader is not None and self.early_stopping.best_state is not None:
                print(f"Training complete. Restoring best model (val_acc={self.early_stopping.best_score:.4f})")
                self.early_stopping.restore_best(self.model)
        
        return history
    
    def _log_progress(self, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict] = None):
        msg = f"Epoch {epoch}"
        for k, v in train_metrics.items():
            msg += f" | train_{k}: {v:.4f}"
        if val_metrics:
            for k, v in val_metrics.items():
                msg += f" | val_{k}: {v:.4f}"
        print(msg)
    
    def save(self, path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_checkpoint(model, optimizer, epoch, path)
