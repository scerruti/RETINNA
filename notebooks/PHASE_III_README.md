# Phase III: SWIR+NIR Binary Burn Segmentation

## Overview

**Objective**: Create a parallel semantic segmentation model using direct spectral indices (SWIR+NIR) aligned to CalFire burn masks, independent from Phase II's severity classification approach.

**Key Differences from Phase II**:
| Aspect | Phase II (8-channel) | Phase III (2-channel) |
|--------|---------------------|----------------------|
| Input | 8 channels (pre + post RGBN) | 2 channels (NIR + SWIR) |
| Processing | Learned change detection | Direct spectral indices |
| Output | 7-class severity | Binary burn/no-burn |
| Labels | Computed RdNBR thresholds | CalFire administrative masks |
| Validation | Multi-class metrics | Binary precision/recall |

**Why Phase III?**
1. **Simpler validation**: Binary labels from official CalFire perimeters (no computed thresholds)
2. **Direct spectral validation**: Test if NIR+SWIR alone can detect burns without pre-fire data
3. **Baseline comparison**: Evaluate if learned change detection (Phase II) improves over fixed indices
4. **Interpretability**: Spectral indices are transparent and easier to understand than learned representations

---

## Project Structure

```
notebooks/
├── III_SWIR_NIR_CalFire_Segmentation.ipynb    ← Main notebook (skeleton with TODOs)
├── cabuaur_data_loader.py                      ← Data loading utilities
├── phase3_config.py                            ← Hyperparameter configuration
├── phase3_losses.py                            ← Custom loss functions
└── PHASE_III_README.md                         ← This file
```

---

## Quick Start

### 1. Data Loading

The notebook expects:
- **Sentinel-2 GeoTIFF files**: NIR (B08) and SWIR (B12) bands
- **CalFire masks**: Fire perimeter polygons rasterized to binary pixel masks
- **Metadata**: Fire event IDs with train/val/test fold assignments

```python
from cabuaur_data_loader import CaBuAurDataLoader

loader = CaBuAurDataLoader(
    data_root=Path('/path/to/sentinel2/data'),
    metadata_path=Path('/path/to/cabuaur_metadata.json')
)

datasets = loader.create_datasets(split_by_event=True)
# Returns: {'train': (nir_swir_tensor, mask_tensor), 'val': ..., 'test': ...}
```

### 2. Model Architecture

Binary U-Net adapted from Phase II:
- Input: 2 channels (NIR, SWIR)
- Output: Single channel with binary logits
- Architecture: Encoder-decoder with skip connections

```python
from III_SWIR_NIR_CalFire_Segmentation import BinaryUNet

model = BinaryUNet(in_channels=2, bilinear=True).to(device)
```

### 3. Loss Function

Select from pre-configured options in `phase3_config.py`:

```python
from phase3_config import LOSS_CONFIG, get_loss_fn
import torch.nn as nn

# Default: BCEWithLogitsLoss with pos_weight for class imbalance
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))

# Or try alternatives:
# - FocalLoss: Adaptive weighting of hard examples
# - DiceLoss: Direct optimization of F1 score
# - TverskyLoss: Control false positive vs false negative trade-off
```

### 4. Training

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Training loop is defined in the notebook
# Iterates over epochs, computes loss, updates weights
```

---

## Configuration Files

### `phase3_config.py`

Quick-switch parameters without editing the notebook:

```python
LOSS_CONFIG = {
    'name': 'bce_with_logits',  # or 'focal', 'dice', 'tversky'
    'params': {'pos_weight': 2.0},
}

TRAINING_CONFIG = {
    'batch_size': 8,
    'learning_rate': 1e-3,
    'num_epochs': 20,
}
```

To iterate on different loss functions:
1. Edit `phase3_config.py`
2. Restart notebook kernel
3. Re-run training cells

### `phase3_losses.py`

Pre-implemented loss functions:
- **BCEWithLogitsLoss**: Standard binary cross-entropy (baseline)
- **FocalLoss**: Focuses on hard-to-classify pixels
- **DiceLoss**: Directly optimizes F1/Dice coefficient
- **TverskyLoss**: Tunable false positive vs false negative trade-off
- **CombinedLoss**: Weighted sum of multiple losses

---

## Data Specification

### Input Tensors

**NIR (B08)**: [N, H, W]
- Sentinel-2 Band 8 (Near-Infrared)
- 20m resolution
- Range: [0, 10000] (raw DN) or [0, 1] (normalized)

**SWIR (B12)**: [N, H, W]
- Sentinel-2 Band 12 (Shortwave-Infrared)
- 20m resolution
- Range: [0, 10000] (raw DN) or [0, 1] (normalized)

**Binary Mask**: [N, H, W]
- 0: No burn (outside CalFire perimeter)
- 1: Burned (inside CalFire perimeter)
- From rasterized CalFire shapefile

### Normalization

Applied in training loop (not dataset, to avoid device conflicts):

```python
# Z-score normalization (computed on training data only)
image_normalized = (image - channel_mean) / (channel_std + 1e-8)
```

Channel statistics computed separately for NIR and SWIR from training split.

---

## Training Loop

### Epoch Structure

```
For each epoch:
  1. Train phase:
     - Iterate over training batches
     - Forward pass: logits = model(normalized_image)
     - Compute loss: loss = criterion(logits, labels)
     - Backward: gradients
     - Optimizer step: update weights
     - Track: loss, accuracy
  
  2. Validation phase:
     - Iterate over validation batches (no gradient)
     - Compute validation loss and metrics
     - Learning rate scheduler step based on val loss
  
  3. Checkpoint:
     - Save if validation improves
     - Log training history
```

### Loss Computation

```python
# Binary classification loss
logits = model(normalized_image)  # [N, H, W]
loss = criterion(logits, labels)  # Scalar
loss.backward()
optimizer.step()
```

---

## Evaluation Metrics

### Per-Image Metrics (Binary Classification)

For each test image:
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP) — "How many detected burns are real?"
- **Recall**: TP / (TP + FN) — "What fraction of real burns did we detect?"
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)

### Pixel-Level Metrics

Aggregate over all test pixels:
- **IoU (Intersection over Union)**: TP / (TP + FP + FN)
- **Confusion Matrix**: [True Neg, False Pos; False Neg, True Pos]

### Expected Performance Baseline

From Phase II insights, expect:
- **Accuracy**: 80-90% (dominated by no-burn class)
- **Burn Recall**: 40-60% (critical metric — must detect actual burns)
- **Burn Precision**: 70-85% (minimize false alarms)
- **F1**: 0.50-0.70

---

## Iterative Development Strategy

### Iteration 1: Baseline (BCE)
- Simple BCEWithLogitsLoss
- Evaluate baseline performance
- Expected: ~31% recall on minority class (from Phase II patterns)

### Iteration 2: Address Imbalance (pos_weight)
- Increase `pos_weight` to up-weight burn pixels
- Test pos_weight ∈ [1.0, 5.0]
- Expected: Improve recall at cost of some precision

### Iteration 3: Alternative Loss (Focal, Dice, Tversky)
- Try FocalLoss for hard-example mining
- Try DiceLoss for direct F1 optimization
- Try TverskyLoss with alpha/beta tuning
- Expected: Better trade-off between recall and precision

### Iteration 4: Data Augmentation
- Tune augmentation parameters
- Test with/without zoom/crop
- Expected: Better generalization to unseen fires

### Iteration 5: Architecture Variations
- Try deeper/shallower encoders
- Test with/without skip connections
- Experiment with batch normalization parameters
- Expected: Small improvements (~2-5%)

---

## Reused Code from Phase II

### Architecture Components
- `DoubleConv`: 3×3 convolution block with BatchNorm and ReLU
- `Down`: Max pooling + DoubleConv for downsampling
- `Up`: Bilinear upsampling + skip connection + DoubleConv
- `UNet`: Full encoder-decoder with skip connections

**Adapted for Phase III**:
- Input channels: 8 → 2 (removed pre-fire and post-fire separation)
- Output channels: 7 (7-class) → 1 (binary logits)
- Loss function: CrossEntropyLoss → BCEWithLogitsLoss

### Data Handling
- Dataset base structure (with augmentation)
- Normalization approach (z-score, computed on training only)
- Augmentation pipeline (flip, rotate, zoom/crop)
- Train/val/test splitting by fire event

**Adapted for Phase III**:
- Removed fold-based splitting (use direct metadata instead)
- Simplified to 2-channel input (NIR, SWIR)
- Single-label regression (binary classification, not 7-class)

### Training Framework
- Epoch-based training loop
- Learning rate scheduling with ReduceLROnPlateau
- Checkpoint saving with metadata
- Training history tracking

---

## Expected Challenges & Solutions

### Challenge 1: Class Imbalance (if burn pixels are minority)

**Symptom**: Model predicts all no-burn, achieves high accuracy but 0% recall

**Solutions**:
1. Increase `pos_weight` in BCEWithLogitsLoss
2. Try FocalLoss to focus on hard pixels
3. Use DiceLoss to optimize F1 directly
4. Threshold adjustment at inference time

### Challenge 2: Spatial Misalignment

**Symptom**: Predictions don't align to actual burn boundaries

**Root Cause**: Rasterization artifacts, coordinate system mismatches

**Solutions**:
1. Verify projection systems match (Sentinel-2 vs CalFire shapefiles)
2. Check GeoTIFF metadata (CRS, transform) before loading
3. Visually inspect rasterized masks vs original polygons
4. Apply morphological operations to clean mask edges

### Challenge 3: Overfitting (if few fires in training)

**Symptom**: Training loss → 0, validation loss increases

**Solutions**:
1. Increase data augmentation probability
2. Add L1/L2 regularization (weight_decay in optimizer)
3. Use early stopping based on validation loss
4. Collect more training fires if possible

---

## File Reference Guide

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `III_SWIR_NIR_CalFire_Segmentation.ipynb` | Main notebook with training loop | `BinaryUNet`, `train_epoch`, `validate` |
| `cabuaur_data_loader.py` | Load Sentinel-2 + CalFire masks | `CaBuAurDataLoader.load_fire_data()` |
| `phase3_config.py` | Hyperparameters & loss selection | `LOSS_CONFIG`, `TRAINING_CONFIG` |
| `phase3_losses.py` | Custom loss implementations | `FocalLoss`, `DiceLoss`, `TverskyLoss` |

---

## Next Steps

1. **Complete data loader**: Implement reading from actual GeoTIFF files and CalFire shapefiles
2. **Test on small subset**: Verify data pipeline with 5-10 fires before full training
3. **Baseline training**: Run BCE loss for 20 epochs, record metrics
4. **Iterate on loss functions**: Test Focal, Dice, Tversky
5. **Hyperparameter tuning**: Optimize pos_weight, learning rate, batch size
6. **Inference & visualization**: Create comparison plots of predictions vs ground truth
7. **Compare to Phase II**: Evaluate if learned change detection (Phase II) beats fixed indices (Phase III)

---

## References

- Sentinel-2 Bands: https://earth.esa.int/eogateway/missions/sentinel-2/description
- CalFire Fire Perimeters: https://www.fire.ca.gov/incidents
- CaBuAur Dataset: Original paper with fire event IDs and coordinates
- Focal Loss: Lin et al. (2017) "Focal Loss for Dense Object Detection"
- Dice Loss: Milletari et al. (2016) "The Dice coefficient for measuring segmentation accuracy"
- Tversky Loss: Salehi et al. (2017) "Tversky loss function for image segmentation"
