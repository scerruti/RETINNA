"""Acreage, agreement with the official perimeter, and a dNBR baseline."""

from dataclasses import asdict, dataclass

import numpy as np
from affine import Affine
from rasterio import features
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from . import config


def acres(mask: np.ndarray, pixel_m: float = config.PIXEL_M) -> float:
    return float(mask.sum()) * pixel_m * pixel_m / config.SQM_PER_ACRE


def rasterize(geom: BaseGeometry, out_shape: tuple[int, int], transform: Affine) -> np.ndarray:
    """geom must already be in the raster's CRS."""
    return features.rasterize(
        [(mapping(geom), 1)], out_shape=out_shape, transform=transform, fill=0, dtype="uint8"
    ).astype(bool)


@dataclass
class Agreement:
    predicted_acres: float
    perimeter_acres: float
    iou: float
    precision: float   # of pixels we call burned, share inside the perimeter
    recall: float      # of pixels inside the perimeter, share we call burned
    f1: float
    acres_outside_perimeter: float
    acres_unburned_inside_perimeter: float

    def to_dict(self):
        return {k: round(v, 4) for k, v in asdict(self).items()}


def compare(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray | None = None) -> Agreement:
    """Pixel agreement between a predicted burn mask and the rasterized perimeter.

    Note: a perimeter is an outer boundary, not a burn map. Unburned islands inside it
    count against recall here even when the model is right about them.
    """
    if valid is not None:
        pred, ref = pred & valid, ref & valid
    tp = np.count_nonzero(pred & ref)
    fp = np.count_nonzero(pred & ~ref)
    fn = np.count_nonzero(~pred & ref)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    px = config.PIXEL_M**2 / config.SQM_PER_ACRE
    return Agreement(
        predicted_acres=acres(pred),
        perimeter_acres=acres(ref),
        iou=tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        precision=p,
        recall=r,
        f1=2 * p * r / (p + r) if p + r else 0.0,
        acres_outside_perimeter=fp * px,
        acres_unburned_inside_perimeter=fn * px,
    )


def nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    nir, swir = nir.astype(np.float32), swir.astype(np.float32)
    return (nir - swir) / (nir + swir + 1e-6)


def dnbr_mask(pre_nir, pre_swir, post_nir, post_swir, threshold=config.DNBR_THRESHOLD):
    """Classic remote-sensing baseline: burned where pre-NBR minus post-NBR exceeds threshold."""
    d = nbr(pre_nir, pre_swir) - nbr(post_nir, post_swir)
    return d > threshold, d


def vectorize(mask: np.ndarray, transform: Affine, min_acres: float = 1.0, simplify_m: float = 15.0):
    """Burn mask -> list of shapely polygons in the raster CRS, dropping specks."""
    min_px_area = min_acres * config.SQM_PER_ACRE
    polys = []
    for geom, val in features.shapes(mask.astype("uint8"), mask=mask, transform=transform):
        if val != 1:
            continue
        g = shape(geom)
        if g.area >= min_px_area:
            polys.append(g.simplify(simplify_m, preserve_topology=True))
    return polys
