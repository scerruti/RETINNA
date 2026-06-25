# Phase II_02: 8-Channel Model Training on Colab

**Updated**: 2026-06-25  
**Status**: Ready for execution  
**Architecture**: 8-channel (Pre+Post RGBN) with z-score normalization + augmentation

---

## Quick Start: Run These Steps on Colab

### Step 1: Run II_01 First (5-10 min)

The updated II_01 notebook now saves pre/post RGBN tensors separately.

1. Open Google Colab: https://colab.research.google.com
2. Upload or open `/notebooks/II_01_spectral_relabeling.ipynb` from your GitHub/Drive
3. Run all cells from top to bottom
4. **New output files saved to Drive**:
   - `pre_rgbn_TIMESTAMP.pt` (shape [424, 4, 512, 512])
   - `post_rgbn_TIMESTAMP.pt` (shape [424, 4, 512, 512])
   - `multi_class_labels_TIMESTAMP.pt` (unchanged, shape [2×424, 512, 512])

**Note the timestamp** — you'll need it for Step 2.

---

### Step 2: Run II_02 (20-30 min training, depending on GPU)

1. Open `/notebooks/II_02_unet_training.ipynb` in Colab
2. In the first data-loading cell, update the file paths to use the timestamps from Step 1:
   ```python
   # Find this cell and update TIMESTAMP to match your II_01 output
   pre_rgbn_path = f"{data_dir}/pre_rgbn_YYYYMMDD_HHMMSS.pt"
   post_rgbn_path = f"{data_dir}/post_rgbn_YYYYMMDD_HHMMSS.pt"
   labels_path = f"{data_dir}/multi_class_labels_YYYYMMDD_HHMMSS.pt"
   ```
3. Run all cells top to bottom
4. **Output files saved to Drive**:
   - `unet_model_20260625_HHMMSS.pt` (checkpoint with normalization stats)
   - Training plots and metrics

---

## What Changed in II_02

### Architecture: 4-Channel → 8-Channel

**Before**: Input was precomputed (Post - Pre) difference, 4 channels
```
Input shape: [4, 512, 512]
Model input: DoubleConv(4, 64)
```

**Now**: Input is Pre_RGBN concatenated with Post_RGBN
```
Input shape: [8, 512, 512]  # [Pre_R, Pre_G, Pre_B, Pre_NIR, Post_R, Post_G, Post_B, Post_NIR]
Model input: DoubleConv(8, 64)
```

**Why**: Model learns flexible change detection patterns instead of hardcoded subtraction. Transfers better to NAIP.

---

### Data Preprocessing: New Steps

#### 1. Z-Score Normalization

Computed from **training samples only** (no validation/test leakage):
```
channel_mean = mean of pre+post channels across training set
channel_std  = std of pre+post channels across training set
Normalization: (image - channel_mean) / (channel_std + 1e-8)
```

Applied to all splits (train, val, test) using training statistics.

**Saved in checkpoint**: `normalization['channel_mean']` and `normalization['channel_std']` (8 values each)

#### 2. Data Augmentation

Applied **only to training set**:
- **Random horizontal flip** (50% chance)
- **Random vertical flip** (50% chance)
- **Random zoom crop** (50% chance): crop to 384×384, resize back to 512×512 (via bilinear interpolation)

Validation and test sets: NO augmentation.

---

### Label Alignment Fix

**Bug Fixed**: II_01 saves labels as shape [2N, 512, 512] (pre + post stacked), but dataset now uses only post-fire labels:
```python
# Old (wrong): idx directly into labels tensor
label = labels[idx]  # Could use pre-fire labels

# New (correct): offset by N to use post-fire labels
label = labels[N + idx]  # Always uses post-fire severity label
```

---

### Class Weights: Fixed

**Before**: Hardcoded `[1.0, 0.01, 1.0, 1.0, 1.0, 1.0, 1.0]` — the 0.01 was a placeholder and wrong.

**Now**: Computed from actual training label distribution:
```python
class_counts = torch.bincount(train_labels.flatten(), minlength=7)
class_weights = 1.0 / (class_counts + 1.0)
class_weights = class_weights / class_weights.sum() * 7  # normalize to mean=1
```

Example output:
```
Class  0 (Unburned):   weight=0.15  (majority class)
Class  1 (Low):        weight=2.30  (rare, upweighted)
Class  2 (Moderate):   weight=1.90
Class  3 (High):       weight=2.10
Class  4 (Extreme):    weight=2.45  (rarest, heavily upweighted)
Class  5 (Water):      weight=0.98
Class  6 (Cloud):      weight=1.12
```

---

## Model Checkpoint Structure

The saved model now includes normalization statistics:

```python
checkpoint = {
    'model_state_dict': {...},           # Same as before
    'normalization': {
        'channel_mean': tensor([...]),   # NEW: 8 channel means
        'channel_std': tensor([...]),    # NEW: 8 channel stds
    },
    'class_weights': tensor([...]),      # NEW: 7 weights computed from training
    'config': {
        'in_channels': 8,                # CHANGED: was 4
        'out_channels': 7,               # Unchanged
    },
    'history': {...},                    # Training history
}
```

---

## Expected Results

Training parameters (same as baseline):
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Loss**: CrossEntropyLoss with class weights
- **Epochs**: 20 (same as before)

Expected improvements:
- ✅ Smoother convergence (due to augmentation)
- ✅ Better generalization (augmentation + z-score norm)
- ✅ More stable across NAIP transfer (learned patterns, not S2-specific absolute values)
- ⚠ May see slightly different accuracy numbers (due to fixed class weights + augmentation)

---

## After Training: Validation

Once II_02 completes on Colab:

1. **Check normalization saved**:
   ```python
   checkpoint = torch.load('unet_model_TIMESTAMP.pt')
   assert 'normalization' in checkpoint
   print(checkpoint['normalization']['channel_mean'])  # Should be 8 values
   ```

2. **Verify in_channels**:
   ```python
   assert checkpoint['config']['in_channels'] == 8
   ```

3. **Review training curves**: Check that loss/accuracy plots show smoother convergence than baseline.

---

## Ready for Phase III

Once training completes:
- ✅ Model accepts 8-channel NAIP inputs (pre + post spectral pairs)
- ✅ Normalization statistics are embedded in checkpoint
- ✅ Class weights are saved (useful if fine-tuning on NAIP needed)
- ✅ Augmentation-trained model should generalize better to NAIP imagery

Next step: Phase III NAIP inference pipeline will:
1. Load NAIP pre/post temporal pairs
2. Concatenate as 8 channels (same format as training)
3. Apply saved normalization statistics
4. Run inference with trained model

---

## Troubleshooting

### File not found: pre_rgbn_TIMESTAMP.pt

Make sure you copied the exact timestamp from II_01 output. The notebook saves with format `YYYYMMDD_HHMMSS`.

### Shape mismatch error in dataloader

Check that:
- `pre_rgbn.pt` shape is [424, 4, 512, 512]
- `post_rgbn.pt` shape is [424, 4, 512, 512]
- `multi_class_labels.pt` shape is [848, 512, 512] (2×424)

### Loss NaN or very high

If loss doesn't decrease:
1. Verify channel_mean and channel_std are not extreme (should be ~0 mean, ~1 std after z-score)
2. Check that augmentation is not distorting the images too much
3. Run with `augment=False` temporarily to isolate the issue

---

## What's Next

After this training run completes:
1. Evaluate on test set (measure per-class accuracy)
2. Compare to baseline (4-channel difference model) — accuracy change is OK as long as test performance is solid
3. Prepare Phase III NAIP pipeline for inference

---

**Questions?** Check `/docs/PHASE_II_02_CHANGE_DETECTION_STRATEGY.md` for architectural rationale.
