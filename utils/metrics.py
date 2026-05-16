import torch
from typing import Dict, List, Optional
from collections import defaultdict


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds.argmax(dim=-1) == labels).float().mean().item()


def f1_score(preds: torch.Tensor, labels: torch.Tensor, num_classes: int, average: str = 'macro') -> float:
    pred_classes = preds.argmax(dim=-1)
    f1_scores = []
    for c in range(num_classes):
        tp = ((pred_classes == c) & (labels == c)).sum().float()
        fp = ((pred_classes == c) & (labels != c)).sum().float()
        fn = ((pred_classes != c) & (labels == c)).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1.item())
    return sum(f1_scores) / len(f1_scores) if average == 'macro' else f1_scores


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'max'):
        """
        Args:
            mode: 'max' for accuracy (higher is better), 'min' for loss (lower is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = -float('inf') if mode == 'max' else float('inf')
        self.best_state = None
    
    def __call__(self, score: float, model=None) -> bool:
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
            matched = score >= self.best_score
        else:
            improved = score < self.best_score - self.min_delta
            matched = score <= self.best_score
        
        if improved:
            self.best_score = score
            self.counter = 0
            if model is not None:
                import copy
                self.best_state = copy.deepcopy(model.state_dict())
            return False
        
        # When score matches best (but not strictly improved), update state
        # to keep the latest model at peak performance
        if matched and model is not None:
            import copy
            self.best_state = copy.deepcopy(model.state_dict())
        
        self.counter += 1
        return self.counter >= self.patience
    
    def restore_best(self, model):
        """Restore best model weights if available."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
    
    def reset(self):
        self.counter = 0
        self.best_score = -float('inf') if self.mode == 'max' else float('inf')
        self.best_state = None


class MetricTracker:
    def __init__(self):
        self.history: Dict[str, List[float]] = defaultdict(list)
    
    def update(self, metrics: Dict[str, float], prefix: str = ''):
        for name, value in metrics.items():
            key = f"{prefix}_{name}" if prefix else name
            self.history[key].append(value)
    
    def get_best(self, metric: str, mode: str = 'min') -> tuple:
        values = self.history.get(metric, [])
        if not values:
            return None, -1
        func = min if mode == 'min' else max
        best_val = func(values)
        return best_val, values.index(best_val)
    
    def get_last(self, metric: str) -> Optional[float]:
        values = self.history.get(metric, [])
        return values[-1] if values else None
    
    def reset(self):
        self.history.clear()
