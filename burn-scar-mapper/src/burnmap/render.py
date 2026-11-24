"""PNG layers for the demo page."""

import numpy as np
from PIL import Image


def _stretch(band: np.ndarray, valid: np.ndarray, lo=2, hi=98) -> np.ndarray:
    v = band[valid] if valid.any() else band.ravel()
    a, b = np.percentile(v, [lo, hi]) if v.size else (0, 1)
    return (np.clip((band - a) / max(b - a, 1e-6), 0, 1) * 255).astype(np.uint8)


def false_color(swir, nir, red, valid) -> Image.Image:
    """SWIR / NIR / Red: burn scars read as dark red-brown, vegetation as green."""
    rgb = np.dstack([_stretch(swir, valid), _stretch(nir, valid), _stretch(red, valid)])
    rgb[~(nir > 0)] = 0
    return Image.fromarray(rgb, "RGB")


def mask_overlay(mask: np.ndarray, color=(255, 60, 0), alpha=150) -> Image.Image:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, alpha)
    return Image.fromarray(rgba, "RGBA")


def probability_overlay(prob: np.ndarray) -> Image.Image:
    """Transparent -> orange -> red as probability rises past 0.5."""
    p = np.nan_to_num(prob)
    rgba = np.zeros((*p.shape, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = (200 * (1 - p)).astype(np.uint8)
    rgba[..., 3] = (np.clip((p - 0.3) / 0.7, 0, 1) * 200).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")
