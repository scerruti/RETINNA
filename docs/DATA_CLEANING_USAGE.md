# Data Validation & Dataset Access

## Overview

CaBuAr provides pre-defined train/val/test splits that are already balanced and validated. Issue #6 validates data quality across these native splits, then provides a simple wrapper for easy PyTorch integration.

## Validation: One-Time Setup

### Run Data Validation (Colab Notebook)

Use `notebooks/03_data_cleaning.ipynb` to validate all splits:

```
# On Colab, run the notebook which will:
1. Load each native split (train, val, test)
2. Check for data quality issues (NaN, zero bands, shape mismatches)
3. Report findings per split
4. Test CaBuArDataset wrapper
```

**Expected output:**
```
CaBuAr Dataset Validation
======================================================================
Train split: 1366 samples
  NaN values: 0
  Zero bands: 0
  Shape mismatches: 0
  All burned tiles: 0
  All unburned tiles: 0

Val split: 547 samples
  [same checks...]

Test split: 547 samples
  [same checks...]

======================================================================
✅ DATASET CLEAN: No corruption detected
```

### What Gets Validated?

- **NaN values**: Missing data in image or mask
- **Zero bands**: Entire spectral band = 0 (dead sensor)
- **Shape mismatch**: Image ≠ [2, 12, 512, 512] or Mask ≠ [1, 512, 512]
- **Empty tiles** (flagged, not removed): 100% burned or 0% burned pixels

## Usage: Loading Data

### Option 1: Direct CaBuAr (Simplest)

```python
from torchgeo.datasets import CaBuAr

dataset = CaBuAr(root='/tmp/cabuaur', split='train', download=True)
sample = dataset[0]
image = sample['image']  # [2, 12, 512, 512]
mask = sample['mask']    # [1, 512, 512]
```

### Option 2: CaBuArDataset Wrapper (Recommended)

```python
from src.dataset import CaBuArDataset

# Load individual split
train_dataset = CaBuArDataset(split='train', root='/tmp/cabuaur')
sample = train_dataset[0]
```

### Option 3: PyTorch DataLoaders (For Training)

```python
from src.dataset import get_dataloaders

dataloaders = get_dataloaders(batch_size=32, num_workers=4)

train_loader = dataloaders['train']
val_loader = dataloaders['val']
test_loader = dataloaders['test']

# Iterate over batches
for batch in train_loader:
    images = batch['image']    # [batch_size, 2, 12, 512, 512]
    masks = batch['mask']      # [batch_size, 1, 512, 512]
    # Train model...
```

## CaBuAr Native Splits

| Split | Samples | Purpose |
|-------|---------|---------|
| **train** | 1,366 | Model training |
| **val** | 547 | Hyperparameter tuning, early stopping |
| **test** | 547 | Final evaluation |
| **all** | 2,460 | Full dataset (for custom analysis) |

## In Notebooks

### 01_data_loading.ipynb
```python
from src.dataset import CaBuArDataset

dataset = CaBuArDataset(split='test', root=cabuaur_path)
print(f"Loaded {len(dataset)} samples")
```

### 02_exploratory_analysis.ipynb
Can use any split; typically 'test' for quick iteration:
```python
dataset = CaBuArDataset(split='test', root=cabuaur_path)
# Run EDA as before...
```

### Training Scripts (Future)
```python
from src.dataset import get_dataloaders

dataloaders = get_dataloaders(batch_size=32)
# Use dataloaders['train'], dataloaders['val'], dataloaders['test']
```

## Why Not Custom Splits?

CaBuAr's native splits are:
- ✅ Pre-validated by dataset authors
- ✅ Balanced and reproducible
- ✅ Standard in published research using CaBuAr
- ✅ No custom code needed

Creating custom splits adds complexity without benefit for this use case.

## Data Files

No additional files are generated. The workflow is:
1. CaBuAr dataset automatically downloads to `{root}/` on first access
2. TorchGeo caches splits metadata
3. `CaBuArDataset` wrapper provides PyTorch-compatible interface

---

**Issue #6 Acceptance Criteria:**
- [x] Data quality validated (no corruption found)
- [x] CaBuAr native splits verified working
- [x] Simple dataset wrapper for PyTorch training
