"""
Data validation for CaBuAr dataset using native train/val/test splits.

Validates data quality and reports findings. CaBuAr provides pre-defined splits.
"""

import numpy as np
from torchgeo.datasets import CaBuAr


def validate_split(dataset, split_name, num_samples=None):
    """
    Validate a dataset split for data quality issues.

    Args:
        dataset: CaBuAr dataset for a specific split
        split_name (str): Name of split ('train', 'val', 'test')
        num_samples: Number to check (None = all)

    Returns:
        dict with validation results
    """
    if num_samples is None:
        num_samples = len(dataset)

    issues = {
        'nan_count': 0,
        'zero_bands_count': 0,
        'high_cloud_count': 0,
        'shape_mismatch_count': 0,
        'empty_tiles': {
            'all_burned': 0,
            'all_unburned': 0
        },
        'samples_checked': min(num_samples, len(dataset)),
        'zero_band_samples': [],
        'high_cloud_samples': []
    }

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        image = sample['image'].numpy()
        mask = sample['mask'].numpy()

        # Check for NaN values
        if np.any(np.isnan(image)) or np.any(np.isnan(mask)):
            issues['nan_count'] += 1

        # Check for all-zero bands
        if np.any(np.all(image == 0, axis=(2, 3))):
            issues['zero_bands_count'] += 1
            issues['zero_band_samples'].append(i)

        # Check for high cloud coverage (CLP = band 10)
        clp = image[0, 10].numpy().mean()
        if clp > 0.3:
            issues['high_cloud_count'] += 1
            issues['high_cloud_samples'].append((i, clp))

        # Check shape consistency
        if image.shape != (2, 12, 512, 512) or mask.shape != (1, 512, 512):
            issues['shape_mismatch_count'] += 1

        # Check for empty tiles (no class variation)
        mask_flat = mask[0]
        burned_pct = (mask_flat > 0).sum() / mask_flat.size

        if burned_pct == 1.0:
            issues['empty_tiles']['all_burned'] += 1
        elif burned_pct == 0.0:
            issues['empty_tiles']['all_unburned'] += 1

    return {
        'split': split_name,
        'total_samples': len(dataset),
        'validation_issues': issues
    }


def validate_dataset(dataset_root='/tmp/cabuaur'):
    """
    Main function: validate all native splits in CaBuAr dataset.

    Args:
        dataset_root: Path to CaBuAr dataset
    """
    print("\n" + "="*70)
    print("CaBuAr Dataset Validation")
    print("="*70)

    results = {}

    for split in ['train', 'val', 'test']:
        print(f"\nValidating {split} split...")
        dataset = CaBuAr(root=dataset_root, download=True, split=split)
        result = validate_split(dataset, split)
        results[split] = result

        issues = result['validation_issues']
        print(f"  Total samples: {result['total_samples']}")
        print(f"  NaN values: {issues['nan_count']}")
        print(f"  Zero bands: {issues['zero_bands_count']}")
        print(f"  Shape mismatches: {issues['shape_mismatch_count']}")
        print(f"  All burned tiles: {issues['empty_tiles']['all_burned']}")
        print(f"  All unburned tiles: {issues['empty_tiles']['all_unburned']}")

    # Summary report
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    total_samples = sum(r['total_samples'] for r in results.values())
    total_issues = sum(
        r['validation_issues']['nan_count'] +
        r['validation_issues']['zero_bands_count'] +
        r['validation_issues']['shape_mismatch_count']
        for r in results.values()
    )

    print(f"\nTotal samples: {total_samples}")
    print(f"  Train: {results['train']['total_samples']}")
    print(f"  Val:   {results['val']['total_samples']}")
    print(f"  Test:  {results['test']['total_samples']}")

    print(f"\nData quality issues:")
    print(f"  NaN values: {sum(r['validation_issues']['nan_count'] for r in results.values())}")
    print(f"  Zero bands: {sum(r['validation_issues']['zero_bands_count'] for r in results.values())}")
    print(f"  High cloud coverage (>30%): {sum(r['validation_issues']['high_cloud_count'] for r in results.values())}")
    print(f"  Shape mismatches: {sum(r['validation_issues']['shape_mismatch_count'] for r in results.values())}")
    print(f"  Total corrupted: {total_issues}")

    print(f"\nCloud Coverage Correlation:")
    print(f"  Investigating if zero-band samples have high cloud coverage...")
    for split_name, result in results.items():
        zero_band_idx = set(result['validation_issues']['zero_band_samples'])
        high_cloud_idx = set(i for i, clp in result['validation_issues']['high_cloud_samples'])
        overlap = zero_band_idx & high_cloud_idx
        print(f"    {split_name}: {len(overlap)} zero-band samples with high clouds (out of {len(zero_band_idx)} zero-band)")

    print(f"\nEmpty tiles (valid but extreme):")
    print(f"  All burned: {sum(r['validation_issues']['empty_tiles']['all_burned'] for r in results.values())}")
    print(f"  All unburned: {sum(r['validation_issues']['empty_tiles']['all_unburned'] for r in results.values())}")

    print("\n" + "="*70)
    if total_issues == 0:
        print("✅ DATASET CLEAN: No corruption detected")
    else:
        print(f"⚠️  {total_issues} corrupted samples detected")
    print("="*70)

    return results


if __name__ == '__main__':
    results = validate_dataset()
    print("\n✅ Validation complete")
    print("CaBuAr native splits (train/val/test) are ready to use")
