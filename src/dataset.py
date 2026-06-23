"""
CaBuAr dataset with train/val/test splits.

Loads cleaned data using split indices from clean_splits.json.
"""

import json
from pathlib import Path
import torch
from torchgeo.datasets import CaBuAr


class CaBuArCleanDataset(torch.utils.data.Dataset):
    """
    CaBuAr dataset with train/val/test splits.

    Filters the base CaBuAr dataset to only return samples from the specified split.
    Requires clean_splits.json (created by data_cleaning.py).

    Example:
        >>> train_dataset = CaBuArCleanDataset(split='train', root='/tmp/cabuaur')
        >>> val_dataset = CaBuArCleanDataset(split='val', root='/tmp/cabuaur')
        >>> test_dataset = CaBuArCleanDataset(split='test', root='/tmp/cabuaur')
    """

    def __init__(self, split='train', root='/tmp/cabuaur', splits_file='data/clean_splits.json'):
        """
        Initialize dataset.

        Args:
            split (str): One of 'train', 'val', 'test'
            root (str): Path to CaBuAr dataset root
            splits_file (str): Path to clean_splits.json
        """
        assert split in ['train', 'val', 'test'], f"split must be one of train/val/test, got {split}"
        self.split = split
        self.root = root
        self.splits_file = splits_file

        # Load base dataset
        self.base_dataset = CaBuAr(root=root, download=True, split='test')

        # Load split indices
        self._load_splits()

    def _load_splits(self):
        """Load split indices from JSON file."""
        splits_path = Path(self.splits_file)

        if not splits_path.exists():
            raise FileNotFoundError(
                f"clean_splits.json not found at {self.splits_file}\n"
                f"Run: python src/data_cleaning.py"
            )

        with open(splits_path, 'r') as f:
            splits_data = json.load(f)

        self.indices = splits_data[self.split]
        self.split_info = splits_data['split_info']

        print(f"✓ Loaded {self.split} split: {len(self.indices)} samples")

    def __len__(self):
        """Return number of samples in this split."""
        return len(self.indices)

    def __getitem__(self, idx):
        """Get sample by index within this split."""
        # idx is relative to this split (0 to len-1)
        # map it to the actual base dataset index
        actual_idx = self.indices[idx]
        return self.base_dataset[actual_idx]

    def get_split_info(self):
        """Return metadata about this split."""
        return {
            'split': self.split,
            'count': len(self.indices),
            'split_info': self.split_info
        }


def get_dataloaders(batch_size=32, num_workers=0, root='/tmp/cabuaur', splits_file='data/clean_splits.json'):
    """
    Convenience function to create train/val/test dataloaders.

    Args:
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        root (str): Path to CaBuAr dataset
        splits_file (str): Path to clean_splits.json

    Returns:
        dict with 'train', 'val', 'test' dataloaders
    """
    train_dataset = CaBuArCleanDataset(split='train', root=root, splits_file=splits_file)
    val_dataset = CaBuArCleanDataset(split='val', root=root, splits_file=splits_file)
    test_dataset = CaBuArCleanDataset(split='test', root=root, splits_file=splits_file)

    return {
        'train': torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        ),
        'val': torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        ),
        'test': torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        ),
        'datasets': {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    }
