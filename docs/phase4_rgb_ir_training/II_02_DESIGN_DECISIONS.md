# Phase II_02: RGB+IR Training — Design Decisions

**Status**: In Progress  
**Date**: 2026-06-24  
**Purpose**: Document architectural and training decisions for multi-class burn detection model

---

## Design Questions & Answers

### 1. Data Augmentation
**Question**: Should we use data augmentation for Phase II_02?

**Answer**: ✅ YES — 90° rotations
- **Rationale**: Dataset size (1,366 samples) is the bottleneck, not domain shift
- **Method**: 90° rotations (0°, 90°, 180°, 270°) are safe for satellite imagery
- **Benefits**:
  - Multiply effective training data: 1,366 → ~5,464 samples (4×)
  - Preserve geographic grid (N/S/E/W alignment)
  - No interpolation artifacts (pixel-perfect rotation)
  - Helps rare classes (Extreme Severity, Water)
- **Implementation**: Randomly choose rotation during training; apply same transform to image AND mask
- **Concern addressed**: Shadows/sun angle bias is mitigated by Sentinel-2's multi-temporal nature

---

### 2. Loss Function
**Question**: What loss function for 7-class imbalanced data?

**Answer**: *(Pending)*
- Options to consider:
  - Cross-entropy with class weighting?
  - Weighted cross-entropy favoring burn classes?
  - Focal loss (emphasize hard examples)?

---

### 3. Model Architecture
**Question**: Keep same U-Net but adjust for 4-channel input and 7-class output?

**Answer**: *(Pending)*
- Considerations:
  - 24-channel → 4-channel input (RGB+IR only, matching NAIP)
  - Binary → 7-class output
  - Same depth/width or adjust?

---

### 4. Training Strategy
**Question**: What hyperparameters and training approach?

**Answer**: *(Pending)*
- Considerations:
  - Same as Phase I? (lr=0.0005, Adam, batch_size=4)
  - Different learning rate for multi-class?
  - Epochs: 20 again? More? Less?
  - Early stopping based on what metric?

---

### 5. Metrics
**Question**: How to measure success on multi-class data?

**Answer**: *(Pending)*
- Considerations:
  - Per-class IoU (especially for rare classes)?
  - Macro F1-score (treats all classes equally)?
  - Weighted F1 (accounts for class imbalance)?
  - Confusion matrix for error analysis?

---

## Summary of Decisions So Far

| Decision | Answer | Rationale |
|----------|--------|-----------|
| **Data Augmentation** | 90° rotations only | Safe for satellite data; 4× dataset multiplication |

---

## Next Steps

1. Decide on loss function (Q2)
2. Decide on model architecture (Q3)
3. Decide on training strategy (Q4)
4. Decide on metrics (Q5)
5. Implement II_02 notebook with all decisions
6. Run training and validation

---

**Owner**: Stephen Cerruti  
**Related**: Phase II_01 (spectral relabeling), Phase II_03 (NAIP transfer)
