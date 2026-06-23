"""
CaBuAr dataset wrapper using native train/val/test splits.

Provides convenient access to CaBuAr's pre-defined splits with PyTorch DataLoader support.
"""

import torch
from torchgeo.datasets import CaBuAr


class CaBuArDataset(torch.utils.data.Dataset):
    """
    Simple wrapper around CaBuAr with native train/val/test splits.

    CaBuAr provides pre-defined, balanced splits. This class makes them
    easy to use with PyTorch training pipelines.

    Normalization: Divides image by 10000 to convert Sentinel-2 L2A DN
    (Digital Numbers, range 0-10000) to reflectance (range 0-1).

    Example:
        >>> train_dataset = CaBuArDataset(split='train')
        >>> val_dataset = CaBuArDataset(split='val')
        >>> test_dataset = CaBuArDataset(split='test')
        >>>
        >>> dataloaders = get_dataloaders(batch_size=32)
        >>> for batch in dataloaders['train']:
        ...     image = batch['image']      # Normalized to [0, 1]
        ...     mask = batch['mask']
    """

    def __init__(self, split='train', root='/tmp/cabuaur', normalize=True):
        """
        Initialize dataset.

        Args:
            split (str): One of 'train', 'val', 'test'
            root (str): Path to CaBuAr dataset root
            normalize (bool): If True, normalize images by dividing by 10000
        """
        assert split in ['train', 'val', 'test'], f"split must be train/val/test, got {split}"
        self.split = split
        self.root = root
        self.normalize = normalize
        self.dataset = CaBuAr(root=root, download=True, split=split)
        print(f"✓ Loaded {self.split} split: {len(self.dataset)} samples")
        if normalize:
            print(f"  Normalization: images / 10000 (Sentinel-2 L2A standard)")

    def __len__(self):
        """Return number of samples in this split."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get sample by index."""
        sample = self.dataset[idx]

        if self.normalize:
            # Normalize image: Sentinel-2 L2A DN (0-10000) → reflectance (0-1)
            sample['image'] = sample['image'].float() / 10000.0

        return sample

    def get_split_info(self):
        """Return metadata about this split."""
        return {
            'split': self.split,
            'count': len(self.dataset)
        }


def get_dataloaders(batch_size=32, num_workers=0, root='/tmp/cabuaur', shuffle_train=True, normalize=True):
    """
    Convenience function to create train/val/test dataloaders.

    Args:
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        root (str): Path to CaBuAr dataset
        shuffle_train (bool): Shuffle training data (default True)
        normalize (bool): Normalize images by dividing by 10000 (default True)

    Returns:
        dict with 'train', 'val', 'test' dataloaders and 'datasets'

    Example:
        >>> dataloaders = get_dataloaders(batch_size=32)
        >>> train_loader = dataloaders['train']
        >>> val_loader = dataloaders['val']
        >>> test_loader = dataloaders['test']
        >>> # Images will be normalized to [0, 1] range
    """
    train_dataset = CaBuArDataset(split='train', root=root, normalize=normalize)
    val_dataset = CaBuArDataset(split='val', root=root, normalize=normalize)
    test_dataset = CaBuArDataset(split='test', root=root, normalize=normalize)

    return {
        'train': torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
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
