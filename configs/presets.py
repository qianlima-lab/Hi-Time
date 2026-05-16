from .base_config import Config, DataConfig, ModelConfig, TrainingConfig


def get_default_config() -> Config:
    """Default config matching paper settings."""
    return Config(
        model=ModelConfig(
            d_model=128,
            codebook_size=64,
            num_quantize_layers=3,
            gpt2_model='./pretrained/gpt2',
        ),
    )


def get_fast_debug_config() -> Config:
    return Config(
        data=DataConfig(batch_size=4),
        model=ModelConfig(d_model=64, n_heads=2, n_layers=1, codebook_size=16, num_quantize_layers=1),
        training=TrainingConfig(
            stage1_epochs=5, stage2_epochs=3, stage3_epochs=5,
            stage1_patience=3, stage3_patience=3,
            num_runs=1, merge_resplit=False,
        ),
    )


# =====================================================================
# Per-dataset optimal presets (L and K selected via grid search on val)
# Paper: L in {1,2,3}, K in {16,32,64,128,256}
# =====================================================================

def _paper_config(codebook_size: int, num_quantize_layers: int, batch_size: int = 32) -> Config:
    """Base config for paper experiments with given K and L."""
    return Config(
        data=DataConfig(batch_size=batch_size),
        model=ModelConfig(
            d_model=128,
            codebook_size=codebook_size,
            num_quantize_layers=num_quantize_layers,
            gpt2_model='./pretrained/gpt2',
        ),
    )


# Best (K, L) per dataset — these need to be determined by grid search.
# Initial values below are reasonable starting points.
DATASET_PRESETS = {
    'ArticularyWordRecognition': (64, 3, 32),   # (K, L, batch_size)
    'FingerMovements':           (64, 3, 32),
    'SpokenArabicDigits':        (128, 3, 64),
    'CharacterTrajectories':     (64, 3, 32),
    'FaceDetection':             (64, 2, 64),
    'InsectWingbeat':            (128, 3, 64),
    'MotorImagery':              (64, 3, 16),
    'SelfRegulationSCP1':        (64, 3, 32),
}


PRESETS = {
    'default': get_default_config,
    'debug': get_fast_debug_config,
}

# Register per-dataset presets
for _ds, (_k, _l, _bs) in DATASET_PRESETS.items():
    # Use lowercase dataset name as preset key
    PRESETS[_ds.lower()] = lambda k=_k, l=_l, bs=_bs: _paper_config(k, l, bs)


def get_preset(name: str) -> Config:
    # Try exact match first, then lowercase
    key = name if name in PRESETS else name.lower()
    if key not in PRESETS:
        available = ', '.join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[key]()
