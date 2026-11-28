"""Sliding-window inference over an arbitrarily sized scene.

The model was trained on 512x512 CaBuAr patches, so a whole fire is cut into overlapping
512 tiles, predicted in batches, and blended back with a tapered weight window so tile
seams don't show up in the burn map.
"""

import numpy as np
import torch

from . import config


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def _blend_window(tile: int, overlap: int) -> np.ndarray:
    """1 in the tile interior, linear ramp to a small floor over the overlap band."""
    ramp = np.ones(tile, dtype=np.float32)
    if overlap > 0:
        edge = np.linspace(0.05, 1.0, overlap, dtype=np.float32)
        ramp[:overlap] = edge
        ramp[-overlap:] = edge[::-1]
    return np.outer(ramp, ramp)


def normalize(nir: np.ndarray, swir: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    x = np.stack([nir, swir]).astype(np.float32)
    return (x - means[:, None, None]) / (stds[:, None, None] + 1e-8)


@torch.no_grad()
def predict_probabilities(
    net: torch.nn.Module,
    x: np.ndarray,
    tile: int = config.TILE,
    overlap: int = config.TILE_OVERLAP,
    batch_size: int = 4,
    device: str = "cpu",
) -> np.ndarray:
    """x: normalized [C, H, W] array -> burn probability [H, W] in [0, 1]."""
    c, h, w = x.shape
    # Pad up to at least one tile; reflect keeps edge statistics plausible.
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph or pw:
        mode = "reflect" if (ph < h and pw < w) else "edge"
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode=mode)
    H, W = x.shape[1:]
    stride = tile - overlap
    window = _blend_window(tile, overlap)
    acc = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)

    coords = [(r, q) for r in _starts(H, tile, stride) for q in _starts(W, tile, stride)]
    for i in range(0, len(coords), batch_size):
        chunk = coords[i : i + batch_size]
        batch = np.stack([x[:, r : r + tile, q : q + tile] for r, q in chunk])
        logits = net(torch.from_numpy(batch).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        for (r, q), p in zip(chunk, probs):
            acc[r : r + tile, q : q + tile] += p * window
            wsum[r : r + tile, q : q + tile] += window
    return (acc / np.maximum(wsum, 1e-8))[:h, :w]
