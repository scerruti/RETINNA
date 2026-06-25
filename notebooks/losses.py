"""
Loss functions for semantic segmentation with class imbalance.

Designed to be importable in Jupyter notebooks and easily configured at runtime.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for multi-class semantic segmentation.

    Optimizes spatial overlap (intersection/union) rather than per-pixel likelihood.
    More robust to class imbalance than CrossEntropyLoss.

    Formula: Loss = 1 - (2 * intersection + eps) / (union + eps)

    Args:
        weight: Optional per-class weights [C]. If None, uniform weighting.
        eps: Small constant for numerical stability. Default 1e-6.
    """

    def __init__(self, weight=None, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model output logits [N, C, H, W]
            targets: Ground truth class indices [N, H, W]

        Returns:
            Scalar loss value
        """
        # Convert logits to probabilities
        probs = F.softmax(inputs, dim=1)  # [N, C, H, W]

        # Convert targets to one-hot encoding
        targets_onehot = F.one_hot(targets, num_classes=probs.shape[1])  # [N, H, W, C]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # [N, C, H, W]

        # Compute intersection and union for each class
        intersection = (probs * targets_onehot).sum(dim=(0, 2, 3))  # [C]
        union = probs.sum(dim=(0, 2, 3)) + targets_onehot.sum(dim=(0, 2, 3))  # [C]

        # Compute Dice coefficient for each class
        dice = (2.0 * intersection + self.eps) / (union + self.eps)  # [C]

        # Apply per-class weights if provided
        if self.weight is not None:
            dice = dice * self.weight

        # Return mean loss
        loss = 1.0 - dice.mean()
        return loss


class TverskyLoss(nn.Module):
    """Tversky Loss for multi-class semantic segmentation.

    Generalizes Dice Loss with tunable false positive/negative tradeoff.
    Useful for optimizing recall vs precision on specific classes.

    Formula: Loss = 1 - TP / (TP + alpha*FP + beta*FN)

    Args:
        alpha: Weight on false positives. Lower alpha penalizes false positives less.
               For recall-focused (fewer false negatives): use alpha < beta
        beta: Weight on false negatives. Lower beta penalizes false negatives less.
              For precision-focused (fewer false positives): use beta < alpha
        weight: Optional per-class weights [C]. If None, uniform weighting.
        eps: Small constant for numerical stability. Default 1e-6.

    Example:
        For Extreme class (want high recall, accept some false positives):
        TverskyLoss(alpha=0.3, beta=0.7) penalizes false negatives 2.3× more
    """

    def __init__(self, alpha=0.5, beta=0.5, weight=None, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.weight = weight
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model output logits [N, C, H, W]
            targets: Ground truth class indices [N, H, W]

        Returns:
            Scalar loss value
        """
        # Convert logits to probabilities
        probs = F.softmax(inputs, dim=1)  # [N, C, H, W]

        # Convert targets to one-hot encoding
        targets_onehot = F.one_hot(targets, num_classes=probs.shape[1])  # [N, H, W, C]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # [N, C, H, W]

        # Compute TP, FP, FN for each class
        # TP: pixels predicted as class AND actually class
        # FP: pixels predicted as class BUT actually not class
        # FN: pixels NOT predicted as class BUT actually class
        TP = (probs * targets_onehot).sum(dim=(0, 2, 3))  # [C]
        FP = (probs * (1 - targets_onehot)).sum(dim=(0, 2, 3))  # [C]
        FN = ((1 - probs) * targets_onehot).sum(dim=(0, 2, 3))  # [C]

        # Compute Tversky index for each class
        tversky = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)  # [C]

        # Apply per-class weights if provided
        if self.weight is not None:
            tversky = tversky * self.weight

        # Return mean loss
        loss = 1.0 - tversky.mean()
        return loss


class FocalLoss(nn.Module):
    """Focal Loss for multi-class semantic segmentation.

    Downweights easy examples and focuses on hard examples.
    Can cause issues with extreme class imbalance (see ITERATION_ANALYSIS.md).

    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: Per-class weights [C]. If None, uniform weighting.
        gamma: Focusing parameter (higher = more aggressive focusing).
               Default 2.0 (100× weight on hard examples)
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model output logits [N, C, H, W]
            targets: Ground truth class indices [N, H, W]

        Returns:
            Scalar loss value
        """
        N, C, H, W = inputs.shape

        # Flatten spatial dimensions
        inputs_flat = inputs.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [N*H*W, C]
        targets_flat = targets.view(-1)  # [N*H*W]

        # Compute log softmax for numerical stability
        log_probs = F.log_softmax(inputs_flat, dim=1)  # [N*H*W, C]

        # Get log probability of true class
        log_p_t = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)  # [N*H*W]

        # Get softmax probabilities for focal weight
        probs = F.softmax(inputs_flat, dim=1)  # [N*H*W, C]
        p_t = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)  # [N*H*W]

        # Cross entropy loss = -log(p_t)
        ce = -log_p_t  # [N*H*W]

        # Apply per-class weighting if alpha provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets_flat)  # [N*H*W]
            ce = ce * alpha_t

        # Focal weight: (1 - p_t)^γ
        # High when model is uncertain/wrong, low when confident/correct
        focal_weight = (1.0 - p_t) ** self.gamma  # [N*H*W]

        # Focal loss = focal_weight * ce
        focal_loss = focal_weight * ce  # [N*H*W]

        return focal_loss.mean()


class CrossEntropyLoss(nn.Module):
    """Wrapper around PyTorch CrossEntropyLoss for consistency.

    Standard pixel-wise cross-entropy loss for semantic segmentation.
    Rewards per-pixel likelihood but vulnerable to class imbalance.

    Args:
        weight: Optional per-class weights [C].
    """

    def __init__(self, weight=None):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model output logits [N, C, H, W]
            targets: Ground truth class indices [N, H, W]

        Returns:
            Scalar loss value
        """
        return self.criterion(inputs, targets)


# Convenience factory function
def get_loss(loss_name, class_weights=None, **kwargs):
    """Get a loss function by name with optional configuration.

    Args:
        loss_name: Name of loss function
            - 'ce': CrossEntropyLoss
            - 'dice': DiceLoss
            - 'tversky': TverskyLoss
            - 'focal': FocalLoss
        class_weights: Optional per-class weights tensor [C]
        **kwargs: Additional arguments passed to loss constructor

    Returns:
        Instantiated loss function module

    Example:
        >>> class_weights = torch.tensor([0.5, 2.0, 1.5, ...])
        >>> loss = get_loss('tversky', class_weights, alpha=0.3, beta=0.7)
    """
    loss_map = {
        'ce': CrossEntropyLoss,
        'dice': DiceLoss,
        'tversky': TverskyLoss,
        'focal': FocalLoss,
    }

    if loss_name not in loss_map:
        raise ValueError(f"Unknown loss: {loss_name}. Choose from {list(loss_map.keys())}")

    loss_class = loss_map[loss_name]

    if loss_name in ['dice', 'tversky', 'focal']:
        return loss_class(weight=class_weights, **kwargs)
    else:  # ce
        return loss_class(weight=class_weights, **kwargs)


# Summary of loss functions and when to use them
LOSS_GUIDE = """
LOSS FUNCTION GUIDE
===================

CrossEntropyLoss (CE):
  - Standard pixel-wise likelihood
  - Pros: Numerically stable, well-understood
  - Cons: Vulnerable to class imbalance, rewards majority class
  - Use when: Class balance is reasonable (<10:1 ratio)

DiceLoss:
  - Optimizes spatial overlap (intersection/union)
  - Pros: Robust to class imbalance, directly optimizes IoU
  - Cons: Less sensitive to per-pixel calibration
  - Use when: Want balanced multi-class performance

TverskyLoss (RECOMMENDED for Extreme class):
  - Generalizes Dice with tunable recall/precision tradeoff
  - Pros: Can prioritize detecting rare classes, smooth gradients
  - Cons: More hyperparameters to tune
  - Use when: Need high recall on specific rare class
  - Example: alpha=0.3, beta=0.7 for Extreme detection
    (penalizes false negatives 2.3× more than false positives)

FocalLoss:
  - Downweights easy examples, focuses on hard examples
  - Pros: Theoretically sound for imbalance
  - Cons: Can create local minima at extreme ratios (see ITERATION_ANALYSIS.md)
  - Use when: Class imbalance moderate (10:1 to 50:1)
  - WARNING: May fail catastrophically at 83:1 ratio

EXTREME CLASS IMBALANCE (83:1 for Low Severity : Extreme):
  Recommended approach: Tversky Loss with alpha < beta
  - alpha controls FP penalty (lower = accept more false positives)
  - beta controls FN penalty (higher = penalize false negatives more)
  - Example config: TverskyLoss(alpha=0.3, beta=0.7, weight=class_weights)
  - This biases model toward detecting Extreme pixels even if uncertain
"""

if __name__ == "__main__":
    print(LOSS_GUIDE)
