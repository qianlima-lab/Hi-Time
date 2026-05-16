#!/usr/bin/env python3
import argparse
import numpy as np

from configs import get_default_config, get_preset
from utils import set_seed, get_device
from utils.data import load_uea_dataset, create_data_loaders
from trainers import ThreeStagePipeline


def run_once(config, X_train, y_train, X_test, y_test, device, seed):
    """Run one full training pipeline with a given seed. Returns test accuracy."""
    set_seed(seed)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test,
        batch_size=config.data.batch_size,
        val_split=0.2,
        num_workers=config.data.num_workers,
        seed=seed,
        merge_and_resplit=config.training.merge_resplit,
        normalize=config.data.normalize
    )
    
    # Update data dimensions from the actual data
    config.data.seq_len = X_train.shape[2]
    config.data.n_vars = X_train.shape[1]
    config.data.num_classes = len(set(y_train))
    
    pipeline = ThreeStagePipeline(config, device)
    models = pipeline.run(train_loader, val_loader)
    
    if test_loader:
        results = pipeline.evaluate(test_loader)
        return results['accuracy']
    return None


def main():
    parser = argparse.ArgumentParser(description='Hi-Time: Multivariate Time Series Classification')
    parser.add_argument('--preset', type=str, default='default', help='Config preset name')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: from config)')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (override config)')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Disable pretrained GPT-2 weights')
    parser.add_argument('--gpt2-model', type=str, default=None,
                        help='GPT-2 model path or name')
    parser.add_argument('--num-runs', type=int, default=None,
                        help='Number of runs with different seeds (default: from config)')
    parser.add_argument('--no-resplit', action='store_true',
                        help='Disable 60/20/20 resplit (use original train/test split)')
    # Hyperparameter overrides for grid search
    parser.add_argument('--codebook-size', type=int, default=None, help='Codebook size K')
    parser.add_argument('--num-quantize-layers', type=int, default=None, help='Quantization depth L')
    args = parser.parse_args()
    
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    try:
        config = get_preset(args.preset)
    except (KeyError, ValueError):
        print(f"Preset '{args.preset}' not found, using default")
        config = get_default_config()
    
    # Apply overrides
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.no_pretrained:
        config.model.use_pretrained_gpt2 = False
        print("GPT-2: Training from scratch (no pretrained weights)")
    else:
        config.model.use_pretrained_gpt2 = True
        if args.gpt2_model:
            config.model.gpt2_model = args.gpt2_model
        print(f"GPT-2: Using pretrained model '{config.model.gpt2_model}'")
    if args.no_resplit:
        config.training.merge_resplit = False
    if args.codebook_size is not None:
        config.model.codebook_size = args.codebook_size
    if args.num_quantize_layers is not None:
        config.model.num_quantize_layers = args.num_quantize_layers
    
    seed = args.seed if args.seed is not None else config.training.seed
    num_runs = args.num_runs if args.num_runs is not None else config.training.num_runs
    
    # Load dataset (raw data, will be split inside run_once)
    X_train, y_train, X_test, y_test = load_uea_dataset(args.dataset, config.data.data_dir)
    
    print(f"\nDataset: {args.dataset}")
    print(f"Total samples: train={X_train.shape[0]}, test={X_test.shape[0]}")
    print(f"Shape: ({X_train.shape[1]} vars, {X_train.shape[2]} time steps)")
    print(f"Classes: {len(set(y_train))}")
    print(f"Resplit 60/20/20: {config.training.merge_resplit}")
    print(f"Codebook size: {config.model.codebook_size}, Quantize layers: {config.model.num_quantize_layers}")
    print(f"Runs: {num_runs}")
    
    accuracies = []
    seeds = [seed + i * 111 for i in range(num_runs)]
    
    for run_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1}/{num_runs} (seed={seed})")
        print(f"{'='*60}")
        
        acc = run_once(config, X_train, y_train, X_test, y_test, device, seed)
        
        if acc is not None:
            accuracies.append(acc)
            print(f"\n>>> Run {run_idx + 1} Test Accuracy: {acc:.4f}")
    
    # Summary
    if accuracies:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS ({len(accuracies)} runs)")
        print(f"{'='*60}")
        for i, acc in enumerate(accuracies):
            print(f"  Run {i+1}: {acc:.4f}")
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"  Mean: {mean_acc:.4f} +/- {std_acc:.4f}")


if __name__ == '__main__':
    main()
