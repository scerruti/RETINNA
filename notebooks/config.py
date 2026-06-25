"""
Runtime configuration for II_02 training notebook.

Modify LOSS_CONFIG to test different loss functions without regenerating the notebook.
"""

# ============================================================================
# LOSS FUNCTION CONFIGURATION
# ============================================================================
# Change these values to test different loss functions at runtime

LOSS_CONFIG = {
    # Which loss function to use: 'ce', 'dice', 'tversky', 'focal'
    'name': 'tversky',

    # Loss-specific hyperparameters (only used for relevant losses)
    'params': {
        # TverskyLoss parameters
        # alpha: FP penalty weight (lower = accept more false positives)
        # beta: FN penalty weight (higher = penalize false negatives more)
        # For high recall on Extreme (recall-biased): alpha < beta
        'alpha': 0.3,
        'beta': 0.7,

        # FocalLoss parameters
        # gamma: focusing parameter (higher = more focus on hard examples)
        'gamma': 2.0,

        # DiceLoss: no special parameters (just weight and eps)
    },

    # Whether to apply per-class weights
    # True: uses computed class_weights (helps with imbalance)
    # False: uniform weighting (baseline)
    'use_class_weights': True,
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    'batch_size': 4,
    'learning_rate': 1e-3,
    'num_epochs': 20,
    'weight_decay': 1e-5,

    # Learning rate scheduler parameters
    'scheduler': {
        'mode': 'min',
        'factor': 0.5,
        'patience': 3,
        'min_lr': 1e-7,
    },

    # Data augmentation
    'augmentation': {
        'enabled': True,
        'flip_h_prob': 0.5,
        'flip_v_prob': 0.5,
        'rotation_prob': 0.25,  # Includes 0° (no rotation)
        'zoom_crop_prob': 0.5,
        'crop_size_min': 384,
        'crop_size_max': 512,
    },

    # Normalization
    'normalization': {
        'method': 'zscore',  # 'zscore' or 'none'
        'compute_from_train_only': True,
    },
}

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

INFERENCE_CONFIG = {
    'batch_size': 4,
    'device': 'cuda',  # 'cuda' or 'cpu'
    'save_predictions': True,
    'compute_metrics': True,
}

# ============================================================================
# PRESETS: Common configurations
# ============================================================================

PRESETS = {
    'iteration_2_baseline': {
        'name': 'ce',
        'params': {},
        'use_class_weights': True,
    },
    'iteration_5_focal': {
        'name': 'focal',
        'params': {'gamma': 2.0},
        'use_class_weights': True,
    },
    'iteration_6_tversky_recall': {
        'name': 'tversky',
        'params': {'alpha': 0.3, 'beta': 0.7},
        'use_class_weights': True,
    },
    'iteration_6_tversky_balanced': {
        'name': 'tversky',
        'params': {'alpha': 0.5, 'beta': 0.5},
        'use_class_weights': True,
    },
    'iteration_6_dice': {
        'name': 'dice',
        'params': {},
        'use_class_weights': True,
    },
}


def get_config_preset(name):
    """Load a preset configuration.

    Args:
        name: Preset name from PRESETS dictionary

    Returns:
        dict with LOSS_CONFIG

    Example:
        >>> loss_cfg = get_config_preset('iteration_6_tversky_recall')
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Choose from {list(PRESETS.keys())}")
    return PRESETS[name]


if __name__ == "__main__":
    print("Available presets:")
    for name, cfg in PRESETS.items():
        print(f"  {name}: {cfg['name']} with {cfg['params']}")
