# Day 4 Training Run: Class Weighting (pos_weight=1.5)

**Date**: 2026-06-24  
**Issue**: #14 (Hyperparameter Tuning - Issue #14, Task 1)  
**Configuration**: 20 epochs, batch size 4, 70 batches per epoch  
**Experiment**: BCEDiceLoss with pos_weight=1.5 (emphasize burn detection)

## Purpose

Test class weighting hypothesis: By penalizing false negatives more (via pos_weight=1.5), the model should improve recall (find more burns) with acceptable precision tradeoff.

**Baseline comparison (from docs/analysis_results/):**
- Baseline IoU: 0.5836 (from test set)
- Baseline Recall: 60.5%
- Baseline Precision: 94.2%
- Baseline F1: 0.737

**Expected improvements:**
- Recall: 60.5% → 65-70% (+5-10%)
- Precision: 94.2% → 91-93% (-1-3%, acceptable)
- F1: 0.737 → 0.745-0.755 (+1%)
- IoU: ~0.58+ (may stay similar or slightly improve)

## Files in This Directory

- `README.md` - This file (experiment description)
- `metrics.json` - Summary statistics (best_iou, best_epoch, final metrics)
- `training.log` - Epoch-level training log (from Colab)
- `confusion_matrix.png` - Test set confusion matrix visualization
- `roc_curve.png` - ROC curve with AUC score
- `threshold_analysis.png` - Metrics vs probability threshold
- (Optional) Training curves - loss and IoU plots

## Tracking & Budget

**Compute Usage:**
- Compute units: 1.33 units
- GPU time: ~1.05 hours
- Peak GPU RAM: 8.8 / 15.0 GB (58.7%)
- Cost-effectiveness: Very good (60% lower than forecast)

See: docs/DAY_4_GPU_TRACKING.md for full compute tracking

## Results (To Be Updated After Training)

### Validation Metrics
| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Best Val IoU | 0.52+ | ? | TBD |
| Best Epoch | 13-15 | ? | TBD |
| Final Recall | 65-70% | ? | TBD |
| Final Precision | 91-93% | ? | TBD |
| Final F1 | 0.745+ | ? | TBD |

### Analysis
- Did class weighting achieve goal? (Improved recall ≥5%?)
- Is precision tradeoff acceptable?
- Should we proceed to Issue #15 (Loss Optimization)?

## References

- Baseline: docs/BASELINE_RESULTS.md
- Baseline analysis: docs/BASELINE_ANALYSIS.md
- Sprint plan: docs/DAY_4_SPRINT_PLAN.md
- Compute tracking: docs/DAY_4_GPU_TRACKING.md
- Exploration: docs/EXPLORATION_HYPERPARAMETER_TUNING.md (class imbalance analysis)
- Code: notebooks/03_training.ipynb (BCEDiceLoss with pos_weight=1.5)

---

**Status**: Experiment ready to run on Colab  
**Next**: Issue #15 (Loss Optimization with different BCE/Dice ratios)
