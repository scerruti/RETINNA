"""Phase III Configuration: SWIR+NIR Binary Burn Segmentation

Configurable parameters for quick iteration on different approaches.
Modify this file to test different loss functions, class weights, and training strategies.
"""

# ============================================================================
# LOSS FUNCTION CONFIGURATION
# ============================================================================

LOSS_CONFIG = {
    'name': 'bce_with_logits',  # Options: 'bce_with_logits', 'focal', 'weighted_bce'
    'params': {
        'pos_weight': 2.0,  # Weight for positive class (burn)
    },
    'use_class_weights': False,  # If True, use computed class balance weights
}

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

TRAINING_CONFIG = {
    'batch_size': 8,
    'learning_rate': 1e-3,
    'num_epochs': 20,
    'weight_decay': 1e-5,
    'optimizer': 'adam',  # Options: 'adam', 'sgd'
    'scheduler': {
        'type': 'reduce_lr_on_plateau',  # Options: 'reduce_lr_on_plateau', 'cosine_annealing'
        'params': {
            'factor': 0.5,
            'patience': 3,
            'min_lr': 1e-7,
        }
    }
}

# ============================================================================
# DATA AUGMENTATION CONFIGURATION
# ============================================================================

AUGMENTATION_CONFIG = {
    'horizontal_flip': 0.5,
    'vertical_flip': 0.5,
    'rotation': True,  # 90/180/270 degrees
    'zoom_crop': {
        'enabled': True,
        'crop_min_ratio': 0.75,  # Minimum 75% of original size
    }
}

# ============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    'model_type': 'binary_unet',
    'in_channels': 2,  # NIR + SWIR
    'bilinear_upsample': True,
    'use_batch_norm': True,
}

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

DATA_CONFIG = {
    'split_by_event': True,  # Split by fire event to prevent leakage
    'val_frac': 0.15,
    'test_frac': 0.15,
    'normalization': 'z_score',  # Options: 'z_score', 'min_max', 'percentile'
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================

METRICS_CONFIG = {
    'compute_per_class': True,
    'track_confusion_matrix': True,
    'save_predictions': True,
}


def get_loss_fn(loss_name: str, **kwargs):
    """Get loss function by name.

    Args:
        loss_name: Name of loss function (from LOSS_CONFIG['name'])
        **kwargs: Additional arguments (pos_weight, class_weights, etc.)

    Returns:
        Loss function callable
    """
    import torch.nn as nn

    if loss_name == 'bce_with_logits':
        pos_weight = kwargs.get('pos_weight', 1.0)
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    elif loss_name == 'focal':
        # Focal loss requires custom implementation
        from phase3_losses import FocalLoss
        gamma = kwargs.get('gamma', 2.0)
        alpha = kwargs.get('alpha', 0.25)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_name == 'weighted_bce':
        # Weighted BCE with per-class weights
        from phase3_losses import WeightedBCELoss
        class_weights = kwargs.get('class_weights', None)
        return WeightedBCELoss(weights=class_weights)

    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


if __name__ == '__main__':
    print("Phase III Configuration")
    print(f"\nLoss: {LOSS_CONFIG['name']}")
    print(f"Params: {LOSS_CONFIG['params']}")
    print(f"\nTraining:")
    print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"  Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"  Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"\nData:")
    print(f"  Split by event: {DATA_CONFIG['split_by_event']}")
    print(f"  Normalization: {DATA_CONFIG['normalization']}")
