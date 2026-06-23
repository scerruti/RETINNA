"""
Data cleaning script for CaBuAr dataset.

Identifies corrupted/empty tiles, removes/flags them, and creates train/val/test splits.
Run this once to generate clean_splits.json for downstream use.
"""

import json
import numpy as np
from pathlib import Path
from torchgeo.datasets import CaBuAr


def identify_corrupted_tiles(dataset, num_samples=None):
    """
    Identify corrupted tiles: NaN, zeros, shape mismatches, extreme values.

    Args:
        dataset: CaBuAr dataset
        num_samples: Number to check (None = all)

    Returns:
        dict with 'corrupted_indices' and 'issues' details
    """
    if num_samples is None:
        num_samples = len(dataset)

    corrupted = []
    issues = {}

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        image = sample['image'].numpy()
        mask = sample['mask'].numpy()

        flags = []

        # Check for NaN values
        if np.any(np.isnan(image)) or np.any(np.isnan(mask)):
            flags.append("contains_nan")

        # Check for all-zero bands (dead sensor)
        if np.any(np.all(image == 0, axis=(2, 3))):
            flags.append("zero_bands")

        # Check shape consistency
        if image.shape != (2, 12, 512, 512) or mask.shape != (1, 512, 512):
            flags.append(f"wrong_shape_img={image.shape}_mask={mask.shape}")

        # Check for extreme value ranges (Sentinel-2 is 16-bit: 0-65535)
        if np.any(image < 0) or np.any(image > 65535):
            flags.append(f"extreme_values_range=[{image.min()},{image.max()}]")

        if flags:
            corrupted.append(i)
            issues[i] = flags

    return {
        'corrupted_indices': corrupted,
        'issues': issues,
        'count': len(corrupted)
    }


def identify_empty_tiles(dataset, num_samples=None):
    """
    Identify tiles with no variation in labels (all burned or all unburned).
    These are useful for QA but may cause issues in stratified sampling.

    Args:
        dataset: CaBuAr dataset
        num_samples: Number to check (None = all)

    Returns:
        dict with 'empty_indices' and class distribution
    """
    if num_samples is None:
        num_samples = len(dataset)

    empty = []
    class_dist = {'all_burned': [], 'all_unburned': []}

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        mask = sample['mask'].numpy()[0]

        burned_pct = (mask > 0).sum() / mask.size

        # Flag tiles with very imbalanced labels
        if burned_pct == 1.0:
            empty.append(i)
            class_dist['all_burned'].append(i)
        elif burned_pct == 0.0:
            empty.append(i)
            class_dist['all_unburned'].append(i)

    return {
        'empty_indices': empty,
        'class_distribution': class_dist,
        'count': len(empty)
    }


def create_splits(dataset, test_indices, val_ratio=0.15, train_ratio=0.70):
    """
    Create train/val/test splits with optional stratification.

    Args:
        dataset: CaBuAr dataset
        test_indices: Indices to exclude (corrupted/empty)
        val_ratio: Fraction of clean data for validation (default 0.15)
        train_ratio: Fraction of clean data for training (default 0.70)
                     remaining goes to test

    Returns:
        dict with train/val/test indices and split info
    """
    # Get clean indices
    all_indices = set(range(len(dataset)))
    clean_indices = sorted(list(all_indices - set(test_indices)))

    n_clean = len(clean_indices)
    n_train = int(n_clean * train_ratio)
    n_val = int(n_clean * val_ratio)

    # Random shuffle for now (stratification could be added later)
    np.random.shuffle(clean_indices)

    train_indices = clean_indices[:n_train]
    val_indices = clean_indices[n_train:n_train + n_val]
    test_indices_final = clean_indices[n_train + n_val:]

    return {
        'train': sorted(train_indices),
        'val': sorted(val_indices),
        'test': sorted(test_indices_final),
        'split_info': {
            'train_count': len(train_indices),
            'val_count': len(val_indices),
            'test_count': len(test_indices_final),
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': 1.0 - train_ratio - val_ratio,
            'total_clean': n_clean
        }
    }


def save_splits(splits, output_path='data/clean_splits.json'):
    """Save splits to JSON file for later use."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"✓ Splits saved to {output_path}")


def clean_dataset(dataset_root='/tmp/cabuaur', output_path='data/clean_splits.json'):
    """
    Main function: load dataset, identify issues, create splits, save config.

    Args:
        dataset_root: Path to CaBuAr dataset
        output_path: Where to save clean_splits.json
    """
    print("\n" + "="*70)
    print("CaBuAr Dataset Cleaning Pipeline")
    print("="*70)

    print(f"\nLoading CaBuAr dataset from {dataset_root}...")
    # Load full dataset (CaBuAr test split is small; use full dataset for cleaning)
    dataset = CaBuAr(root=dataset_root, download=True, split='all')
    print(f"✓ Loaded {len(dataset)} total samples")

    # Check all samples for corruption
    print("\nScanning for corrupted tiles...")
    corrupted = identify_corrupted_tiles(dataset)
    print(f"✓ Found {corrupted['count']} corrupted tiles")
    if corrupted['count'] > 0:
        print(f"  Issues: {corrupted['issues']}")

    # Check for empty tiles
    print("\nScanning for empty/imbalanced tiles...")
    empty = identify_empty_tiles(dataset)
    print(f"✓ Found {empty['count']} empty tiles")
    print(f"  All burned: {len(empty['class_distribution']['all_burned'])}")
    print(f"  All unburned: {len(empty['class_distribution']['all_unburned'])}")

    # Combine corrupted + empty as "exclude" set
    exclude_indices = set(corrupted['corrupted_indices'] + empty['empty_indices'])

    # Create splits
    print("\nCreating train/val/test splits...")
    splits = create_splits(dataset, exclude_indices, val_ratio=0.15, train_ratio=0.70)

    print(f"\n" + "="*70)
    print("SPLIT SUMMARY")
    print("="*70)
    print(f"Total samples: {len(dataset)}")
    print(f"Corrupted (removed): {corrupted['count']}")
    print(f"Empty/imbalanced (flagged): {empty['count']}")
    print(f"Clean samples: {splits['split_info']['total_clean']}")
    print(f"\nTrain: {splits['split_info']['train_count']} ({splits['split_info']['train_ratio']*100:.0f}%)")
    print(f"Val:   {splits['split_info']['val_count']} ({splits['split_info']['val_ratio']*100:.0f}%)")
    print(f"Test:  {splits['split_info']['test_count']} ({splits['split_info']['test_ratio']*100:.0f}%)")
    print("="*70)

    # Save splits
    save_splits(splits, output_path)

    return splits


if __name__ == '__main__':
    # Run with default paths
    splits = clean_dataset()
    print("\n✅ Data cleaning complete")
    print(f"Use clean_splits.json with CaBuArCleanDataset for train/val/test loading")
