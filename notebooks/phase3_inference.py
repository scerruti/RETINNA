"""Phase III Inference: Load checkpoint and make predictions on new data.

Usage:
    checkpoint = torch.load('phase3_model_checkpoint.pt')
    predictor = Phase3Predictor(checkpoint)
    predictions = predictor.predict(nir_band, red_band)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


class BinaryUNet(nn.Module):
    """Binary U-Net for burn/no-burn segmentation."""

    def __init__(self, in_channels=2, bilinear=True):
        super().__init__()
        factor = 2 if bilinear else 1

        # Encoder
        self.inc = self.DoubleConv(in_channels, 64)
        self.down1 = self.Down(64, 128)
        self.down2 = self.Down(128, 256)
        self.down3 = self.Down(256, 512)
        self.down4 = self.Down(512, 1024 // factor)

        # Decoder
        self.up1 = self.Up(1024, 512 // factor, bilinear)
        self.up2 = self.Up(512, 256 // factor, bilinear)
        self.up3 = self.Up(256, 128 // factor, bilinear)
        self.up4 = self.Up(128, 64, bilinear)

        # Output
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    @staticmethod
    def DoubleConv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def Down(in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            BinaryUNet.DoubleConv(in_channels, out_channels)
        )

    @staticmethod
    def Up(in_channels, out_channels, bilinear=True):
        if bilinear:
            up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv = BinaryUNet.DoubleConv(in_channels, out_channels)
            return nn.Sequential(up, conv)
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                BinaryUNet.DoubleConv(in_channels // 2, out_channels)
            )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outc(x).squeeze(1)
        return x


class Phase3Predictor:
    """Load Phase III checkpoint and make predictions."""

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        """
        Args:
            checkpoint_path: Path to phase3_model_checkpoint.pt
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create model
        self.model = BinaryUNet(in_channels=2).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Get normalization parameters
        self.channel_means = checkpoint['normalization']['channel_means'].to(self.device)
        self.channel_stds = checkpoint['normalization']['channel_stds'].to(self.device)

        # Get metrics for reference
        self.metrics = checkpoint.get('metrics', {})

        print(f"✓ Model loaded from {checkpoint_path}")
        print(f"  Channel means: {self.channel_means.cpu().numpy()}")
        print(f"  Channel stds: {self.channel_stds.cpu().numpy()}")
        print(f"  Test Recall (burn detection rate): {self.metrics.get('test_recall', 'N/A')}")

    def predict(self, nir_band: np.ndarray, red_band: np.ndarray,
                threshold: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Make predictions on NIR and Red bands.

        Args:
            nir_band: [H, W] NIR band (B08, values typically 0-10000 or 0-1)
            red_band: [H, W] Red band (B04, values typically 0-10000 or 0-1)
            threshold: Threshold for binary classification (default 0.0 = unbiased)
                      Increase to reduce false positives
                      Decrease to increase recall (detect more burns)

        Returns:
            {
                'logits': [H, W] raw model outputs,
                'probabilities': [H, W] sigmoid(logits),
                'predictions': [H, W] binary predictions (0/1),
                'ndvi': [H, W] computed NDVI index,
            }
        """
        # Convert to torch tensors
        nir = torch.from_numpy(nir_band).float().to(self.device)
        red = torch.from_numpy(red_band).float().to(self.device)

        # Compute NDVI
        ndvi = (nir - red) / (nir + red + 1e-8)

        # Stack to 2-channel image
        image = torch.stack([ndvi, nir], dim=0)  # [2, H, W]
        image = image.unsqueeze(0)  # [1, 2, H, W]

        # Normalize
        mean = self.channel_means.view(2, 1, 1)
        std = self.channel_stds.view(2, 1, 1)
        image = (image - mean) / (std + 1e-8)

        # Predict
        with torch.no_grad():
            logits = self.model(image).squeeze(0)  # [H, W]
            probs = torch.sigmoid(logits)
            predictions = (logits > threshold).long()

        return {
            'logits': logits.cpu().numpy(),
            'probabilities': probs.cpu().numpy(),
            'predictions': predictions.cpu().numpy(),
            'ndvi': ndvi.cpu().numpy(),
        }

    def predict_batch(self, nir_batch: torch.Tensor, red_batch: torch.Tensor,
                     threshold: float = 0.0) -> torch.Tensor:
        """
        Batch prediction for efficiency.

        Args:
            nir_batch: [B, H, W] NIR bands
            red_batch: [B, H, W] Red bands
            threshold: Classification threshold

        Returns:
            [B, H, W] Binary predictions
        """
        nir_batch = nir_batch.float().to(self.device)
        red_batch = red_batch.float().to(self.device)

        # Compute NDVI
        ndvi = (nir_batch - red_batch) / (nir_batch + red_batch + 1e-8)

        # Stack
        images = torch.stack([ndvi, nir_batch], dim=1)  # [B, 2, H, W]

        # Normalize
        mean = self.channel_means.view(1, 2, 1, 1)
        std = self.channel_stds.view(1, 2, 1, 1)
        images = (images - mean) / (std + 1e-8)

        # Predict
        with torch.no_grad():
            logits = self.model(images)  # [B, H, W]
            predictions = (logits > threshold).long()

        return predictions


if __name__ == '__main__':
    # Example usage
    checkpoint_path = '/Users/scerruti/RETINNA/phase3_model_checkpoint.pt'

    if Path(checkpoint_path).exists():
        predictor = Phase3Predictor(checkpoint_path, device='cpu')

        # Create synthetic test data
        h, w = 256, 256
        nir = np.random.randint(0, 10000, (h, w)).astype(np.float32)
        red = np.random.randint(0, 10000, (h, w)).astype(np.float32)

        # Make predictions
        result = predictor.predict(nir, red)

        print(f"\n✓ Predictions made:")
        print(f"  Logits shape: {result['logits'].shape}")
        print(f"  Burn pixels: {result['predictions'].sum()} / {result['predictions'].size}")
        print(f"  Burn percentage: {100 * result['predictions'].sum() / result['predictions'].size:.1f}%")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Train the model first by running the notebook.")
