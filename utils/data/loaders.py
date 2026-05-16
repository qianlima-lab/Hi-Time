import numpy as np
import os
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

from .datasets import TimeSeriesDataset


def load_uea_dataset(dataset_name: str, data_path: str = './data/datasets') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset_dir = os.path.join(data_path, dataset_name)
    
    # Try loading from local npy files first
    npy_train_x = os.path.join(dataset_dir, 'X_train.npy')
    npy_train_y = os.path.join(dataset_dir, 'y_train.npy')
    npy_test_x = os.path.join(dataset_dir, 'X_test.npy')
    npy_test_y = os.path.join(dataset_dir, 'y_test.npy')
    
    if all(os.path.exists(f) for f in [npy_train_x, npy_train_y, npy_test_x, npy_test_y]):
        X_train = np.load(npy_train_x)
        y_train = np.load(npy_train_y)
        X_test = np.load(npy_test_x)
        y_test = np.load(npy_test_y)
        
        # Convert string labels to integers if needed
        if y_train.dtype.kind in ['U', 'S', 'O']:
            unique_labels = np.unique(np.concatenate([y_train, y_test]))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            y_train = np.array([label_map[y] for y in y_train])
            y_test = np.array([label_map[y] for y in y_test])
        
        print(f"Loaded {dataset_name} from local npy files")
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}, Classes: {len(np.unique(y_train))}")
        return X_train, y_train, X_test, y_test
    
    # Try using aeon library
    try:
        from aeon.datasets import load_classification
        X_train, y_train = load_classification(dataset_name, split='train')
        X_test, y_test = load_classification(dataset_name, split='test')
        
        if X_train.ndim == 3:
            X_train = X_train.transpose(0, 2, 1)
            X_test = X_test.transpose(0, 2, 1)
        
        label_map = {label: i for i, label in enumerate(np.unique(y_train))}
        y_train = np.array([label_map[y] for y in y_train])
        y_test = np.array([label_map[y] for y in y_test])
        
        print(f"Loaded {dataset_name} from aeon")
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, y_train, X_test, y_test
    except ImportError:
        raise ImportError(f"Dataset {dataset_name} not found locally and aeon is not installed. "
                         f"Please install aeon: pip install aeon")


def normalize_data(X_train: np.ndarray, X_val: np.ndarray = None,
                   X_test: np.ndarray = None) -> tuple:
    """Per-variable z-score normalization using train statistics.
    
    Input shape: (N, V, T) — normalize along N and T for each variable V.
    """
    # Compute mean/std per variable from training data: shape (V,)
    mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1, V, 1)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8   # (1, V, 1)
    
    X_train = (X_train - mean) / std
    if X_val is not None:
        X_val = (X_val - mean) / std
    if X_test is not None:
        X_test = (X_test - mean) / std
    
    return X_train, X_val, X_test


def create_data_loaders(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
                        batch_size: int = 32, val_split: float = 0.2,
                        num_workers: int = 0, seed: int = 42,
                        merge_and_resplit: bool = False,
                        normalize: bool = True) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Create data loaders.
    
    Args:
        merge_and_resplit: If True, merge train+test then resplit 60/20/20
                          (paper protocol). If False, use original split with
                          val carved from train (legacy behavior).
        normalize: If True, apply per-variable z-score normalization.
    """
    if merge_and_resplit and X_test is not None and y_test is not None:
        # Paper protocol: merge all data, then split 60/20/20 with stratification
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        
        # First split: 60% train, 40% temp
        X_train_new, X_temp, y_train_new, y_temp = train_test_split(
            X_all, y_all, test_size=0.4, random_state=seed, stratify=y_all
        )
        # Second split: 50% of 40% = 20% val, 20% test
        X_val, X_test_new, y_val, y_test_new = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
        )
        
        print(f"  Resplit (60/20/20): train={X_train_new.shape[0]}, val={X_val.shape[0]}, test={X_test_new.shape[0]}")
        
        if normalize:
            X_train_new, X_val, X_test_new = normalize_data(X_train_new, X_val, X_test_new)
        
        train_dataset = TimeSeriesDataset(X_train_new, y_train_new)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        test_dataset = TimeSeriesDataset(X_test_new, y_test_new)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader, test_loader
    
    # Legacy behavior: original train/test split, val carved from train
    if normalize:
        X_train, _, X_test = normalize_data(X_train, None, X_test)
    
    train_dataset = TimeSeriesDataset(X_train, y_train)
    
    val_loader = None
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    test_loader = None
    if X_test is not None and y_test is not None:
        test_dataset = TimeSeriesDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
