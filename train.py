"""
Training script for U-Net burn scar detection model.

Trains on CaBuAr dataset with BCE+Dice loss for class imbalance handling.
Implements checkpoint saving, validation monitoring, and early stopping.

Based on PA3 pattern from starter.py but adapted for:
- 12-channel Sentinel-2 multispectral input (vs PA3's 3-channel RGB)
- Binary burn detection with severe class imbalance (90% unburned)
- Satellite imagery precision requirements
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
from pathlib import Path

from src.unet import UNet
from src.dataset import get_dataloaders
from src.device_utils import get_device, move_to_device


class BCEDiceLoss(nn.Module):
    """Hybrid loss combining BCE and Dice for class imbalance."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, 2, H, W] logits (not yet softmax/sigmoid)
            targets: [B, 1, H, W] binary mask (0 or 1)
        """
        # BCE loss (per-pixel binary cross-entropy)
        bce_loss = self.bce(predictions, targets.float())

        # Dice loss (IoU-based, good for imbalanced data)
        probs = torch.softmax(predictions, dim=1)  # [B, 2, H, W]
        burned_prob = probs[:, 1:2]  # [B, 1, H, W] - probability of burned class

        intersection = (burned_prob * targets).sum()
        union = (burned_prob + targets).sum()
        dice_loss = 1 - (2 * intersection + 1e-8) / (union + 1e-8)

        # Combined loss
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss


def compute_iou(predictions, targets, threshold=0.5):
    """Compute Intersection over Union for burned class."""
    with torch.no_grad():
        probs = torch.softmax(predictions, dim=1)  # [B, 2, H, W]
        burned_pred = (probs[:, 1] > threshold).float()  # [B, H, W]
        targets_binary = targets.squeeze(1).float()  # [B, H, W]

        intersection = (burned_pred * targets_binary).sum()
        union = (burned_pred + targets_binary).sum() - intersection
        iou = intersection / (union + 1e-8)

    return iou.item()


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        batch = move_to_device(batch, device)
        images = batch['image']  # [B, 2, 12, 512, 512] (bi-temporal)
        masks = batch['mask']    # [B, 1, 512, 512]

        # Flatten timesteps into channels: [B, 2, 12, 512, 512] -> [B, 24, 512, 512]
        B, T, C, H, W = images.shape
        images = images.view(B, T * C, H, W)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)  # [B, 2, 512, 512]
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}: Loss = {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = move_to_device(batch, device)
            images = batch['image']  # [B, 2, 12, 512, 512]
            masks = batch['mask']    # [B, 1, 512, 512]

            # Flatten timesteps into channels
            B, T, C, H, W = images.shape
            images = images.view(B, T * C, H, W)

            outputs = model(images)
            loss = criterion(outputs, masks)
            iou = compute_iou(outputs, masks)

            total_loss += loss.item()
            total_iou += iou

    avg_loss = total_loss / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    return avg_loss, avg_iou


def save_checkpoint(model, optimizer, epoch, val_iou, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_iou': val_iou,
    }

    # Save latest checkpoint
    latest_path = checkpoint_dir / 'latest.pth'
    torch.save(checkpoint, latest_path)
    print(f"  Saved latest checkpoint: {latest_path}")

    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / 'best.pth'
        torch.save(checkpoint, best_path)
        print(f"  ★ Saved best checkpoint: {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load checkpoint from file."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint.get('val_iou', 0.0)


def main(args):
    """Main training loop."""
    print("\n" + "="*70)
    print("U-Net Burn Detection Training")
    print("="*70)

    # Setup device
    device = get_device()

    # Create output directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = vars(args)
    config_path = checkpoint_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

    # Load model and data
    print("\nLoading model and data...")
    model = UNet(in_channels=24, out_channels=2).to(device)
    print(f"Model parameters: {model.get_parameter_count():,}")

    dataloaders = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=True
    )

    print(f"Train samples: {len(dataloaders['datasets']['train'])}")
    print(f"Val samples: {len(dataloaders['datasets']['val'])}")
    print(f"Test samples: {len(dataloaders['datasets']['test'])}")

    # Setup loss, optimizer
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Optional: load checkpoint
    start_epoch = 0
    best_val_iou = 0.0
    if args.resume and Path(args.resume).exists():
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch, best_val_iou = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from epoch {start_epoch}, best IoU: {best_val_iou:.4f}")

    # Training loop
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)

        # Train
        train_loss = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device
        )
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss, val_iou = validate(
            model, dataloaders['val'], criterion, device
        )
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        # Checkpoint
        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
            print(f"★ New best validation IoU: {best_val_iou:.4f}")

        save_checkpoint(
            model, optimizer, epoch + 1, val_iou,
            checkpoint_dir, is_best=is_best
        )

        # Early stopping (optional)
        if args.early_stopping and epoch > 0:
            if val_iou < best_val_iou - args.early_stopping_patience:
                print(f"\nEarly stopping triggered (no improvement for {args.early_stopping_patience} epochs)")
                break

    print("\n" + "="*70)
    print("Training complete!")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train U-Net for burn scar detection'
    )

    # Training hyperparameters
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size (default: 32; reduce to 16/8 if OOM with 12 channels)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.0005,
        help='Learning rate for Adam optimizer (default: 0.0005, range: 0.0001-0.001)'
    )
    parser.add_argument(
        '--num-workers', type=int, default=0,
        help='Number of data loading workers (default: 0; increase on CPU)'
    )

    # Checkpointing
    parser.add_argument(
        '--checkpoint-dir', type=str, default='checkpoints',
        help='Directory to save checkpoints (default: checkpoints/)'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from'
    )

    # Early stopping
    parser.add_argument(
        '--early-stopping', action='store_true',
        help='Enable early stopping'
    )
    parser.add_argument(
        '--early-stopping-patience', type=int, default=10,
        help='Early stopping patience in epochs (default: 10)'
    )

    args = parser.parse_args()

    print(f"\nTraining Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    main(args)
