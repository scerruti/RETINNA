# Model Iteration History: II_02 U-Net Training

## Iteration 1: Baseline (4-Channel Difference Model)
**Status**: Previous baseline, replaced

- Input: 4-channel (Post - Pre) difference images
- Loss: Weighted CrossEntropyLoss
- Val Accuracy: 88.36%
- Test Accuracy: Unknown (need to run inference)
- Notes: Older architecture, provides baseline for comparison

---

## Iteration 2: 8-Channel with Z-Score Normalization & Augmentation
**Status**: COMPLETED - UNDERPERFORMING

**Date**: 2026-06-25  
**Model Checkpoint**: `unet_model_20260625_201329.pt`

### Architecture Changes
- Input: 8 channels (Pre_RGBN + Post_RGBN concatenated)
- Normalization: Z-score (channel_mean/std from training set)
- Augmentation: Horizontal flip, vertical flip, random rotation, zoom/crop 384→512
- Loss Function: CrossEntropyLoss with computed class weights

### Performance Results

**Test Set Metrics**:
- Overall Pixel Accuracy: 81.58%
- Macro-Averaged IoU: 0.2535
- Macro-Averaged Precision: 0.29
- Macro-Averaged Recall: 0.38

**Per-Class Performance**:
```
Class              Accuracy  IoU     Recall  Precision  F1
Low Severity       86.44%    0.848   86%     98%        0.92
Cloud              90.56%    0.573   91%     61%        0.73
Water              56.34%    0.321   56%     43%        0.49
Extreme            31.19%    0.033   31%     4%         0.06  ← CRITICAL FAILURE
Unburned           0.00%     0.000   0%      0%         0.00
Moderate           0.00%     0.000   0%      0%         0.00
High               0.00%     0.000   0%      0%         0.00
```

### Root Cause Analysis

**Problem**: Severe class imbalance in training data
- Low Severity: 83% of pixels (14.8M)
- Extreme: 0.9% of pixels (162k) ← Most important for burn severity
- Water: 8% of pixels (1.4M)
- Moderate: 2.4% of pixels (429k)
- Cloud: 6% of pixels (997k)
- High: 0.2% of pixels (28k)
- Unburned: 0.002% of pixels (128)

**Effect**: Model learned to predict "Low Severity" as default safe strategy, ignoring minority classes entirely.

**Metrics Interpretation**:
- 81.58% overall accuracy is **misleading** (success on majority class)
- 31% Extreme recall is **unacceptable** for operational burn severity assessment
- 0% accuracy on Unburned, Moderate, High shows complete failure on minorities
- Weighted F1 of 0.84 hides macro F1 of 0.31 (true multi-class performance)

### Why Current Approach Failed

**Loss Function Issue**:
CrossEntropyLoss with computed class weights:
```python
class_weights = 1.0 / (class_counts + 1.0)  # Inverse frequency
class_weights = class_weights / class_weights.mean()  # Normalize
```

Problem: Even with weighting, the loss function treats all misclassifications equally. 
When Low Severity dominates, a correct Low Severity prediction gets rewarded heavily,
and incorrect Low Severity predictions don't get punished enough. The model converges
to a local optimum: "always predict Low Severity."

**Not a Problem With**:
- Architecture: 8-channel design is sound
- Augmentation: Already implemented, not the bottleneck
- Learning rate: 1e-3 with ReduceLROnPlateau is well-tuned
- Normalization: Z-score is correct
- Training: Model converged properly (smooth loss curves)

---

## Iteration 3: 8-Channel with Focal Loss
**Status**: IN PROGRESS

**Date**: 2026-06-25 (update)  
**Model**: Will retrain with same architecture, different loss

### Changes Made

**Loss Function**: Switched to Focal Loss
```python
# OLD (COMMENTED OUT):
# criterion = nn.CrossEntropyLoss(weight=class_weights)
# Reason: Weighted cross-entropy insufficient for 83:1 class imbalance

# NEW:
# Focal Loss: designed for extreme class imbalance
# - Down-weights easy negatives (Low Severity majority)
# - Focuses training on hard positives (rare Extreme pixels)
# - γ=2.0: 100× more weight to misclassified Extreme
# - α=class_weights: maintains per-class weighting
```

### Expected Improvements

**Primary Goal**: Improve Extreme class detection
- Target Extreme Recall: 50-70% (from 31%)
- Target Extreme F1: 0.30-0.50 (from 0.06)

**Secondary**: Don't degrade other classes
- Keep Low Severity recall ≥ 80%
- Keep Cloud recall ≥ 85%

### Hyperparameters (Unchanged)
- Learning rate: 1e-3 with ReduceLROnPlateau
- Augmentation: Flip, rotate, zoom/crop (unchanged)
- Batch size: 4
- Epochs: 20
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)

### Success Criteria
1. ✅ Extreme recall ≥ 50%
2. ✅ Extreme F1 ≥ 0.25
3. ✅ Low Severity recall ≥ 80%
4. ✅ Overall macro F1 ≥ 0.40

---

## Options for Further Improvement (If Focal Loss Insufficient)

### Option A: Data Rebalancing (Medium Effort)
**Cost**: Retrain (~30 min GPU)  
**Risk**: Low  
**Expected Gain**: +10-20% Extreme recall  

Oversample minority classes during training:
- Keep Low Severity at 70% (down from 83%)
- Boost Extreme to 20% (up from 0.9%)
- Rebalance others to 10%

Implementation: Use WeightedRandomSampler in DataLoader

### Option B: Hybrid Loss Function (Medium Effort)
**Cost**: Implement + retrain (~1 hour)  
**Risk**: Medium  
**Expected Gain**: +15-25% Extreme recall  

Combine multiple loss functions:
- Focal Loss for imbalance handling
- Weighted Dice Loss for segmentation-specific metrics
- Weighted combination: 0.7×Focal + 0.3×Dice

Rationale: Dice loss naturally handles imbalance by focusing on intersection.

### Option C: Advanced Loss Functions (High Effort)
**Cost**: Implement + retrain (~2 hours)  
**Risk**: Medium  
**Expected Gain**: +20-30% Extreme recall  

Try specialized segmentation losses:
- **Tversky Loss**: Tunable precision/recall trade-off (β parameter)
- **Lovász-Softmax**: Optimizes IoU directly for segmentation
- **Boundary Loss**: Adds spatial weight to class boundaries

Best for: Ensuring Extreme pixels are detected even if imperfectly classified

### Option D: Class-Specific Data Augmentation (Low Effort)
**Cost**: Modify augmentation logic + retrain (~45 min)  
**Risk**: Very low  
**Expected Gain**: +5-10% Extreme recall  

Apply stronger augmentation to Extreme samples:
- More aggressive zoom/crop (300-400px min crop)
- More rotation angles
- Add elastic deformation
- Less augmentation on Low Severity to preserve majority class patterns

Best combined with: Focal Loss or rebalancing

### Option E: Loss Function Hyperparameter Tuning (Low Effort)
**Cost**: Multiple retrains (~2 hours)  
**Risk**: Very low  
**Expected Gain**: +5-15% Extreme recall  

If using Focal Loss, tune:
- γ (gamma): Higher = more focus on hard cases. Try: 1.0, 2.0, 3.0
- α (alpha): Class-specific weighting. Could override with manual values

Standard: γ=2.0, α=class_weights  
Aggressive: γ=3.0, α=increased_extreme_weight

---

## Recommended Path Forward

### Phase 1 (Immediate)
1. ✅ Implement Focal Loss (this commit)
2. ⏳ Retrain II_02 with focal loss
3. Run inference (II_04) on new model
4. Compare to Iteration 2 baseline

### Phase 2 (If Extreme recall < 50%)
1. Implement data rebalancing (WeightedRandomSampler)
2. Retrain with Focal Loss + rebalancing
3. Run inference
4. Compare results

### Phase 3 (If still < 60%)
1. Try Tversky Loss (focus on IoU)
2. Or: Hybrid Focal + Dice Loss
3. Optionally add class-specific augmentation

---

## Tracking

| Iteration | Date | Loss Function | Extreme Recall | Overall Accuracy | Status |
|-----------|------|---------------|-----------------|------------------|--------|
| 1 (4ch) | 2026-06-25 | CE + weights | Unknown | Unknown | Baseline (need inference) |
| 2 (8ch) | 2026-06-25 | CE + weights | 31% | 81.6% | **UNDERPERFORMING** |
| 3 (8ch+focal) | 2026-06-25 | Focal | TBD | TBD | IN PROGRESS |
| 4 (8ch+focal+rebal) | TBD | Focal | TBD | TBD | CONDITIONAL |
| 5 (8ch+hybrid) | TBD | Focal+Dice | TBD | TBD | CONDITIONAL |

---

## References

**Focal Loss Paper**: Lin et al. (2017) "Focal Loss for Dense Object Detection"  
**Implementation**: PyTorch community implementations available

**Why Focal Loss for Segmentation**:
- Originally designed for object detection (99% background imbalance)
- Directly applicable to semantic segmentation with class imbalance
- Proven in medical image segmentation (similar 80:1+ ratios)
