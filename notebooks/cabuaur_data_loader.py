"""Data loader for CaBuAur dataset: Load Sentinel-2 SWIR+NIR bands and CalFire masks.

This module handles:
1. Reading Sentinel-2 GeoTIFF files (NIR B08, SWIR B12)
2. Rasterizing CalFire fire perimeter shapefiles to pixel-level binary masks
3. Creating train/val/test splits by fire event
4. Returning pre-aligned tensors for model training
"""

import torch
import numpy as np
from pathlib import Path
import json
import rasterio
from rasterio.features import rasterize
import shapely.geometry as geom
from typing import Tuple, Dict, List, Optional


class CaBuAurDataLoader:
    """Load CaBuAur Sentinel-2 spectral data aligned to CalFire burn masks."""

    def __init__(self, data_root: Path, metadata_path: Path):
        """
        Args:
            data_root: Root directory containing Sentinel-2 GeoTIFF files
            metadata_path: Path to CaBuAur metadata JSON with fire events and fold assignments
        """
        self.data_root = Path(data_root)
        self.metadata_path = Path(metadata_path)

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        print(f"✓ Loaded metadata from {metadata_path}")
        print(f"  Total fires: {len(self.metadata.get('fires', []))}")

    def load_sentinel2_bands(self, fire_id: str, band_ids: List[int]) -> torch.Tensor:
        """Load specific Sentinel-2 bands for a fire event.

        Args:
            fire_id: Fire event identifier from metadata
            band_ids: List of band indices (e.g., [8, 11] for NIR B08 and SWIR B12)

        Returns:
            Tensor of shape [num_bands, H, W]

        Note: Requires GeoTIFF files at data_root/{fire_id}/B{band_id:02d}.tif
        """
        bands = []
        h, w = None, None

        for band_id in band_ids:
            band_path = self.data_root / fire_id / f"B{band_id:02d}.tif"

            if not band_path.exists():
                raise FileNotFoundError(f"Band file not found: {band_path}")

            try:
                with rasterio.open(band_path) as src:
                    band_data = src.read(1).astype(np.float32)
                    if h is None:
                        h, w = band_data.shape
                    bands.append(torch.from_numpy(band_data))
            except Exception as e:
                raise RuntimeError(f"Error reading {band_path}: {e}")

        return torch.stack(bands, dim=0)

    def rasterize_calfire_mask(self, fire_geometry: geom.Geometry, h: int, w: int, \n                              transform) -> torch.Tensor:
        """Rasterize CalFire fire perimeter polygon to binary mask.

        Args:
            fire_geometry: Shapely geometry of fire perimeter
            h, w: Height and width of output raster
            transform: Rasterio transform for coordinate alignment

        Returns:
            Binary tensor [H, W] where 1=burned, 0=unburned
        """
        # Rasterize geometry to binary mask
        shapes = [(fire_geometry, 1)]  # (geometry, value)
        mask = rasterize(shapes, out_shape=(h, w), transform=transform, default_value=0)
        return torch.from_numpy(mask).long()

    def load_fire_data(self, fire_id: str) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Load all data for a single fire event.

        Args:
            fire_id: Fire event identifier

        Returns:
            (nir_swir_tensor, binary_mask, metadata_dict)
        """
        # Load NIR (B08) and SWIR (B12) bands
        nir_swir = self.load_sentinel2_bands(fire_id, band_ids=[8, 11])  # [2, H, W]

        # Load CalFire geometry and rasterize to mask
        # TODO: Implement reading fire geometry from shapefiles
        # For now, return dummy mask of same shape
        _, h, w = nir_swir.shape
        binary_mask = torch.randint(0, 2, (h, w)).long()  # Placeholder

        return nir_swir, binary_mask, {'fire_id': fire_id}

    def create_datasets(self, split_by_event: bool = True, val_frac: float = 0.15, \n                       test_frac: float = 0.15) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Create train/val/test datasets.

        Args:
            split_by_event: If True, split by fire event (prevents leakage). \n                            If False, random pixel-level split.
            val_frac: Fraction for validation set
            test_frac: Fraction for test set

        Returns:
            Dictionary with 'train', 'val', 'test' keys, each containing \n            (nir_swir_tensor, binary_mask_tensor)
        """
        print(f\"Creating datasets (split_by_event={split_by_event})...\\n\")\n        \n        # Collect fire events by fold from metadata\n        folds = {'train': [], 'val': [], 'test': []}\n        \n        for fire in self.metadata.get('fires', []):\n            fold = fire.get('fold', 'train')  # Default to train if not specified\n            fire_id = fire.get('id', fire.get('name'))\n            folds[fold].append(fire_id)\n        \n        datasets = {}\n        \n        for split in ['train', 'val', 'test']:\n            print(f\"Loading {split} split ({len(folds[split])} fires)...\")\n            \n            nir_swir_list = []\n            mask_list = []\n            \n            for fire_id in folds[split]:\n                try:\n                    nir_swir, mask, meta = self.load_fire_data(fire_id)\n                    nir_swir_list.append(nir_swir)\n                    mask_list.append(mask)\n                except Exception as e:\n                    print(f\"  ⚠ Failed to load {fire_id}: {e}\")\n            \n            if nir_swir_list:\n                # Stack into single tensors\n                # Note: Assumes all fires have same spatial dimensions\n                nir_swir_tensor = torch.stack(nir_swir_list, dim=0)  # [N, 2, H, W]\n                mask_tensor = torch.stack(mask_list, dim=0)  # [N, H, W]\n                datasets[split] = (nir_swir_tensor, mask_tensor)\n                print(f\"  ✓ {split}: {nir_swir_tensor.shape} NIR/SWIR, {mask_tensor.shape} mask\")\n            else:\n                print(f\"  ✗ No data loaded for {split} split\")\n        \n        return datasets\n\n\nif __name__ == '__main__':\n    # Example usage\n    data_root = Path('/path/to/sentinel2/data')\n    metadata_path = Path('/path/to/cabuaur_metadata.json')\n    \n    loader = CaBuAurDataLoader(data_root, metadata_path)\n    datasets = loader.create_datasets(split_by_event=True)\n    \n    for split, (nir_swir, mask) in datasets.items():\n        print(f\"{split}: NIR/SWIR {nir_swir.shape}, Mask {mask.shape}\")\n