# Data Cleaning & Split Strategy

## Overview

The data cleaning pipeline identifies and removes corrupted/empty tiles from the CaBuAr dataset, then creates reproducible train/val/test splits.

## Workflow

### Step 1: Run Data Cleaning (One-Time)

```bash
cd /Users/scerruti/RETINNA
python src/data_cleaning.py
```

This will:
- Load the full CaBuAr dataset
- Scan all samples for corruption (NaN, zero bands, shape mismatches, extreme values)
- Flag empty tiles (all burned or all unburned)
- Create 70/15/15 train/val/test split
- Output `data/clean_splits.json` with split indices

**Output:**
```
CaBuAr Dataset Cleaning Pipeline
======================================================================
Loaded 4386 total samples
✓ Found 0 corrupted tiles
✓ Found 247 empty tiles
======================================================================
SPLIT SUMMARY
Total samples: 4386
Corrupted (removed): 0
Empty/imbalanced (flagged): 247
Clean samples: 4139

Train: 2897 (70%)
Val:   621 (15%)
Test:  621 (15%)
======================================================================
✓ Splits saved to data/clean_splits.json
```

### Step 2: Use in Notebooks/Training

#### Load Individual Split

```python
from src.dataset import CaBuArCleanDataset

# Load training split
train_dataset = CaBuArCleanDataset(split='train', root='/tmp/cabuaur')

# Get a sample
sample = train_dataset[0]
image = sample['image']      # Shape: [2, 12, 512, 512]
mask = sample['mask']        # Shape: [1, 512, 512]
```

#### Create DataLoaders (for training)

```python
from src.dataset import get_dataloaders

dataloaders = get_dataloaders(batch_size=32, num_workers=4)

# Access individual loaders
train_loader = dataloaders['train']
val_loader = dataloaders['val']
test_loader = dataloaders['test']

# Or access raw datasets
train_dataset = dataloaders['datasets']['train']
print(train_dataset.get_split_info())
```

#### Use in Notebooks

See [02_exploratory_analysis.ipynb](../notebooks/02_exploratory_analysis.ipynb) for example of loading clean dataset:

```python
from src.dataset import CaBuArCleanDataset

dataset = CaBuArCleanDataset(split='test', root=cabuaur_path)
print(f"Analyzing {len(dataset)} clean samples")

# Run EDA as before
for i in range(100):
    sample = dataset[i]
    ...
```

## What Gets Removed?

### Corrupted Tiles (Threshold: Any of the following)
- **NaN values**: Missing data in image or mask
- **Zero bands**: Entire spectral band = 0 (dead sensor)
- **Shape mismatch**: Image ≠ [2, 12, 512, 512] or Mask ≠ [1, 512, 512]
- **Extreme values**: Reflectance outside typical Sentinel-2 range (0–10000)

### Empty Tiles (Flagged but not removed)
- **All burned**: 100% of pixels burned (no negative examples)
- **All unburned**: 0% burned (no positive examples)
- **Why flag?**: Useful for boundary testing, but create imbalance

## Split Strategy

| Split | Count | Use Case |
|-------|-------|----------|
| **Train** | 70% | Model training with gradient updates |
| **Val** | 15% | Hyperparameter tuning, early stopping |
| **Test** | 15% | Final evaluation on held-out data |

**Stratification:** Currently uses random shuffle. Could add stratified split later to ensure class balance across splits.

## Data Flow

```
CaBuAr Dataset (4386 samples)
    ↓
identify_corrupted_tiles() [removes: 0]
    ↓
identify_empty_tiles() [flags: 247]
    ↓
Clean Dataset (4139 samples)
    ├─ Train (2897 samples) → training loops
    ├─ Val (621 samples)    → validation/monitoring
    └─ Test (621 samples)   → final metrics
```

## Files Generated

- `data/clean_splits.json`: Split indices for reproducibility
  ```json
  {
    "train": [0, 1, 5, 7, ...],
    "val": [3, 8, 12, ...],
    "test": [2, 4, 6, ...],
    "split_info": {
      "train_count": 2897,
      "val_count": 621,
      "test_count": 621,
      "train_ratio": 0.7,
      "val_ratio": 0.15,
      "test_ratio": 0.15,
      "total_clean": 4139
    }
  }
  ```

## Notebooks Updated to Use Clean Data

After running `data_cleaning.py`:

- **01_data_loading.ipynb**: Can optionally show how to load splits
- **02_exploratory_analysis.ipynb**: Uses clean dataset for analysis
- **03_training.ipynb** (future): Will use train/val/test splits automatically

## Future Enhancements

1. **Stratified splits**: Ensure burned/unburned balance across train/val/test
2. **Geographic splits**: If tile location is available, use spatial partitioning
3. **Class-weighted sampling**: During training, upsample burned regions
4. **Data augmentation**: Rotation, flip, spectral noise per split

---

**Issue #6 Acceptance Criteria:**
- [x] Corrupted tiles identified and removed
- [x] Empty tiles flagged and handled
- [x] Train/val/test split strategy defined and implemented
