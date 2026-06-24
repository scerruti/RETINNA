# Class Weighting Inference & Analysis Checklist

**Training Model**: Epoch 16 (Best Val IoU: 0.5609)  
**Test Set**: 644 samples  
**Issue**: #14 (Hyperparameter Tuning)

---

## What to Save from Colab

### From 04_inference.ipynb

**Must save (critical data):**
1. ✓ `predictions.pt` 
   - Contains: predictions, targets, images for all test samples
   - Location: Save to Google Drive
   - Purpose: Ground truth for metrics computation
   - Format: PyTorch tensor file with dict keys: predictions, targets, images

**Optional (already have from baseline):**
2. Sample predictions visualization 
   - Can skip if similar to baseline (3-4 sample tiles with predictions)
   - Only save if showing different behavior (recall improvement visible)

### From 05_analysis.ipynb (Run on Test Set with predictions.pt)

**Must save (quantitative results):**
1. ✓ `metrics.json`
   - Fields: threshold, accuracy, precision, recall, f1_score, iou, roc_auc, specificity, sensitivity, confusion matrix values, test_samples, total_pixels, burned_pixels_ground_truth, burned_pixels_predicted
   - Location: inference_results/metrics.json
   - Purpose: Compare to baseline test metrics

2. ✓ `confusion_matrix.png`
   - Location: inference_results/confusion_matrix.png
   - Purpose: Visual representation of TP/FP/FN/TN

3. ✓ `roc_curve.png`
   - Location: inference_results/roc_curve.png
   - Purpose: Class discrimination across thresholds

4. ✓ `threshold_analysis.png`
   - Location: inference_results/threshold_analysis.png
   - Purpose: Show IoU/precision/recall tradeoff

**Optional but informative:**
5. `error_analysis.png` (sample FP/FN)
   - Location: inference_results/error_analysis.png
   - Purpose: Understand where class weighting helped/hurt

---

## Directory Structure

```
day4_class_weighting_epoch20/
├── README.md
├── INFERENCE_CHECKLIST.md        ← This file
├── training.log
├── training_loss_curves.png
├── validation_iou_curves.png
└── inference_results/            ← New folder
    ├── metrics.json              ← Must have
    ├── confusion_matrix.png      ← Must have
    ├── roc_curve.png            ← Must have
    ├── threshold_analysis.png    ← Must have
    └── error_analysis.png        ← Nice to have
```

---

## How to Save to Colab

**For metrics.json and PNG files:**
```python
# Download from Colab or copy from inference_results folder
# Then move to this directory on your local machine
```

**For predictions.pt:**
```python
# In Colab, it's already backed up to Drive by 04_inference.ipynb
# Should be at: /content/drive/MyDrive/RETINNA_checkpoints/predictions.pt
```

---

## Comparison Checklist (After Saving)

- [ ] metrics.json exists in inference_results/
- [ ] Compare test IoU to validation IoU (should be similar)
- [ ] Check if class weighting improved recall on test set
- [ ] Verify precision didn't drop too much
- [ ] Confirm confusion matrix shows improvement (more TP, fewer FN)

---

## Decision Point After Analysis

**Go to Issue #15 if:**
- ✓ Class weighting improved test IoU (confirmed hypothesis)
- ✓ Recall improved with acceptable precision tradeoff
- ✓ Budget remaining (91.76 units - yes!)

**Consider alternatives if:**
- ❌ Test metrics degraded from validation (generalization issue)
- ❌ Recall didn't improve significantly
- ❌ Precision dropped too much

---

**Next**: Run 04_inference.ipynb → Run 05_analysis.ipynb → Download results → Update this checklist → Proceed to Issue #15

