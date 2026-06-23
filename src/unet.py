"""
U-Net Architecture for Semantic Segmentation

Custom PyTorch implementation of U-Net for binary burn scar detection on Sentinel-2 imagery.
- Input: [B, 12, 512, 512] (2 timesteps × 11 bands + cloud probability)
- Output: [B, 2, 512, 512] (binary segmentation: unburned/burned)

Reference: U-Net: Convolutional Networks for Biomedical Image Segmentation
(Ronneberger et al., 2015) https://arxiv.org/abs/1505.04597
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Double convolution block: Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """Encoder block: DoubleConv → MaxPool (downsampling)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_out = self.conv(x)
        pool_out = self.pool(conv_out)
        return conv_out, pool_out  # Return both for skip connection


class DecoderBlock(nn.Module):
    """Decoder block: TransposeConv (upsampling) → Concat skip → DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Upsample: in_channels → out_channels, 2× spatial dimensions
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concatenation with skip, we have out_channels*2
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        # Upsample
        x = self.up(x)

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)

        # Double convolution
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    U-Net for semantic segmentation of burn scars in satellite imagery.

    Architecture:
    - Encoder: 4 levels of downsampling (feature extraction)
    - Bottleneck: deepest level (context aggregation)
    - Decoder: 4 levels of upsampling with skip connections (precise localization)

    Args:
        in_channels (int): Number of input channels (default 24 for bi-temporal Sentinel-2: 2 timesteps × 12 bands)
        out_channels (int): Number of output classes (default 2 for binary segmentation)
        features (list): Channel progression [64, 128, 256, 512]
    """

    def __init__(self, in_channels=24, out_channels=2, features=None):
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder (downsampling path)
        self.encoder1 = EncoderBlock(in_channels, features[0])  # 12 → 64
        self.encoder2 = EncoderBlock(features[0], features[1])  # 64 → 128
        self.encoder3 = EncoderBlock(features[1], features[2])  # 128 → 256
        self.encoder4 = EncoderBlock(features[2], features[3])  # 256 → 512

        # Bottleneck (deepest level)
        self.bottleneck = DoubleConv(features[3], features[3] * 2)  # 512 → 1024

        # Decoder (upsampling path with skip connections)
        self.decoder4 = DecoderBlock(features[3] * 2, features[3])  # 1024 → 512
        self.decoder3 = DecoderBlock(features[3], features[2])  # 512 → 256
        self.decoder2 = DecoderBlock(features[2], features[1])  # 256 → 128
        self.decoder1 = DecoderBlock(features[1], features[0])  # 128 → 64

        # Final output layer: 1×1 convolution to get class predictions
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through U-Net.

        Args:
            x: Input tensor [batch_size, 12, 512, 512]

        Returns:
            Output tensor [batch_size, 2, 512, 512]
        """
        # Encoder: downsampling with skip connections saved
        skip1, x = self.encoder1(x)  # [B, 64, 512, 512] → skip, [B, 64, 256, 256]
        skip2, x = self.encoder2(x)  # [B, 128, 256, 256] → skip, [B, 128, 128, 128]
        skip3, x = self.encoder3(x)  # [B, 256, 128, 128] → skip, [B, 256, 64, 64]
        skip4, x = self.encoder4(x)  # [B, 512, 64, 64] → skip, [B, 512, 32, 32]

        # Bottleneck
        x = self.bottleneck(x)  # [B, 1024, 32, 32]

        # Decoder: upsampling with skip connections concatenated
        x = self.decoder4(x, skip4)  # [B, 512, 64, 64]
        x = self.decoder3(x, skip3)  # [B, 256, 128, 128]
        x = self.decoder2(x, skip2)  # [B, 128, 256, 256]
        x = self.decoder1(x, skip1)  # [B, 64, 512, 512]

        # Final output: class predictions
        x = self.final_conv(x)  # [B, 2, 512, 512]

        return x

    def get_parameter_count(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test instantiation and forward pass
    model = UNet(in_channels=12, out_channels=2)
    print(f"U-Net created successfully")
    print(f"Total parameters: {model.get_parameter_count():,}")

    # Test forward pass with dummy input
    x = torch.randn(1, 12, 512, 512)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (1, 2, 512, 512), "Output shape mismatch!"
    print("✓ Forward pass test passed")
