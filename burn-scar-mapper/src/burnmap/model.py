"""Binary U-Net, copied from notebooks/III_SWIR_NIR_CalFire_Segmentation.ipynb.

Module names must stay identical to the notebook (inc.double_conv..., down1.maxpool_conv...,
up1.conv...) or the trained state_dict won't load.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
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


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x2.size() != x1.size():
            x1 = F.pad(x1, (0, x2.size(3) - x1.size(3), 0, x2.size(2) - x1.size(2)))
        return self.conv(torch.cat([x2, x1], dim=1))


class BinaryUNet(nn.Module):
    """2-channel (NIR B08 + SWIR B12) -> 1 logit per pixel. Output shape [N, H, W]."""

    def __init__(self, in_channels=2, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

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
        return self.outc(x).squeeze(1)


@dataclass
class LoadedModel:
    net: BinaryUNet
    channel_means: np.ndarray  # [2], raw-DN scale
    channel_stds: np.ndarray   # [2]
    metrics: dict = field(default_factory=dict)
    device: str = "cpu"
    name: str = ""  # checkpoint file name; recorded in results so stale caches can be detected


def load_checkpoint(path: str | Path, device: str = "cpu") -> LoadedModel:
    """Load a checkpoint saved by scripts/save_checkpoint_cell.py.

    Expected keys: model_state_dict, normalization.{channel_means, channel_stds}.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    in_ch = state["inc.double_conv.0.weight"].shape[1]
    if in_ch != 2:
        raise ValueError(
            f"{path} has {in_ch} input channels; Burn Scar Mapper expects the 2-channel "
            "NIR+SWIR Phase III model. (Phase II checkpoints are 4 or 8 channels.)"
        )
    net = BinaryUNet(in_channels=2, bilinear=True)
    net.load_state_dict(state)
    net.to(device).eval()
    norm = ckpt["normalization"]
    to_np = lambda t: np.asarray(t.cpu() if torch.is_tensor(t) else t, dtype=np.float32)  # noqa: E731
    return LoadedModel(
        net=net,
        channel_means=to_np(norm["channel_means"]),
        channel_stds=to_np(norm["channel_stds"]),
        metrics=ckpt.get("metrics", {}),
        device=device,
        name=Path(path).name,
    )
