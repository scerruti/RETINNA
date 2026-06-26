"""Custom loss functions for Phase III binary burn segmentation.

Implements several loss functions for addressing class imbalance in burn detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Paper: Lin et al. (2017) "Focal Loss for Dense Object Detection"

    Downweights easy examples and focuses on hard misclassified pixels.
    Useful when burn class is minority.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor in [0, 1] for class imbalance
            gamma: Exponent factor for down-weighting easy examples
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, H, W] predicted logits (from model output)
            targets: [N, H, W] binary targets (0 or 1)

        Returns:
            Scalar loss value
        """
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            logits.view(-1), targets.view(-1).float(), reduction='none'
        )

        # Get probabilities
        p = torch.sigmoid(logits.view(-1))
        p_t = torch.where(targets.view(-1).bool(), p, 1 - p)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        alpha_t = torch.where(targets.view(-1).bool(),
                             torch.full_like(p, self.alpha),
                             torch.full_like(p, 1 - self.alpha))

        # Focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        return focal_loss.mean()


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss.

    Allows per-class weighting for imbalanced datasets.
    """

    def __init__(self, weights: torch.Tensor = None):
        """
        Args:
            weights: [2] tensor with weights for (no-burn, burn) classes
        """
        super().__init__()
        if weights is None:
            weights = torch.tensor([1.0, 1.0])
        self.register_buffer('weights', weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, H, W] predicted logits
            targets: [N, H, W] binary targets

        Returns:
            Scalar loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            logits.view(-1), targets.view(-1).float(), reduction='none'
        )

        # Weight by class
        class_weights = torch.where(targets.view(-1).bool(),
                                   torch.full_like(bce_loss, self.weights[1]),
                                   torch.full_like(bce_loss, self.weights[0]))

        weighted_loss = class_weights * bce_loss
        return weighted_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss (F1 Loss) for semantic segmentation.

    Directly optimizes Dice coefficient / F1 score, good for imbalanced data.
    More stable than cross-entropy for minority classes.
    """

    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth: Smoothing constant to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, H, W] predicted logits
            targets: [N, H, W] binary targets

        Returns:
            Scalar loss value (1 - Dice coefficient)
        """
        # Get probabilities
        probs = torch.sigmoid(logits)

        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1).float()

        # Compute Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_coeff


class TverskyLoss(nn.Module):
    """Tversky Loss for semantic segmentation.

    Generalization of Dice loss that allows control over false positives vs false negatives.
    Useful for imbalanced data.

    When alpha=beta=0.5, Tversky = Dice
    When alpha > beta, penalizes false positives more
    When beta > alpha, penalizes false negatives more
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
        """
        Args:
            alpha: Weight for false positive (default 0.5 = equal to Dice)
            beta: Weight for false negative (default 0.5 = equal to Dice)
            smooth: Smoothing constant
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, H, W] predicted logits
            targets: [N, H, W] binary targets

        Returns:
            Scalar loss value (1 - Tversky index)
        """
        # Get probabilities
        probs = torch.sigmoid(logits)

        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1).float()

        # Compute true positive, false positive, false negative
        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1 - targets_flat)).sum()
        fn = ((1 - probs_flat) * targets_flat).sum()

        # Tversky index
        tversky_index = (tp + self.smooth) / \
                       (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky_index


class CombinedLoss(nn.Module):
    """Combine multiple loss functions for better performance.

    Example: BCEWithLogitsLoss + DiceLoss
    """

    def __init__(self, losses: list, weights: list):
        """
        Args:
            losses: List of loss functions
            weights: List of weights for each loss
        """
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, H, W] predicted logits
            targets: [N, H, W] binary targets

        Returns:
            Weighted sum of losses
        """
        total_loss = 0.0
        for loss, weight in zip(self.losses, self.weights):
            total_loss += weight * loss(logits, targets)
        return total_loss


if __name__ == '__main__':
    # Test losses
    import torch

    batch_size = 4
    h, w = 256, 256

    logits = torch.randn(batch_size, h, w)
    targets = torch.randint(0, 2, (batch_size, h, w))

    # Test each loss
    losses_to_test = [
        ('BCE', nn.BCEWithLogitsLoss()),
        ('Focal', FocalLoss(alpha=0.25, gamma=2.0)),
        ('Dice', DiceLoss()),
        ('Tversky', TverskyLoss(alpha=0.5, beta=0.5)),
    ]

    for name, loss_fn in losses_to_test:
        loss_val = loss_fn(logits, targets)
        print(f"{name:15s}: {loss_val.item():.6f}")
