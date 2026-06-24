# Exploration: Hyperparameter Tuning & Real-World Deployment Mismatch

**Date**: 2026-06-24  
**Context**: Day 4 Sprint Planning (Issues #14-15)  
**Author**: Stephen Cerruti with Claude  
**Status**: Pre-experiment investigation (informs Day 4 design decisions)

---

## Overview

This document explores critical insights about hyperparameter tuning that emerged during Day 4 planning. Rather than treat tuning as purely empirical parameter search, we investigate **deeper architectural and deployment mismatches** that may limit improvements.

**Key Questions Addressed:**
1. How does Dice loss interact with extreme class imbalance?
2. What is the information cost of dropping pre-fire imagery?
3. How does our training setup (25% burned) differ from real-world scenarios (0.1-1% burned)?
4. What does this mean for production deployment?

---

## Part 1: Dice Loss & Class Imbalance Analysis

### The Dice Loss Formula

```
Dice = 2 * intersection / (pred_sum + target_sum)
```

Where:
- `intersection` = pixels both predicted AND ground-truth burned
- `pred_sum` = total pixels predicted as burned
- `target_sum` = total ground-truth burned pixels
- `denominator` = union-like measure (not true union, but similar)

### Class Imbalance Problem

**In CaBuAr dataset:**
- Burned pixels: 25% (1.19M per 4.87M pixel image)
- Unburned pixels: 75% (3.68M per image)

**Naive prediction problem:**
```python
# What if model predicts all unburned?
pred_sum = 0  (no burned predictions)
target_sum = 1.19M  (true burns)
intersection = 0
Dice = 0 / 1.19M = 0  ✓ Correctly punished

# But for unburned class:
pred_sum = 4.87M  (all predictions unburned)
target_sum = 3.68M  (true unburned)
intersection = 3.68M  (all true unburned pixels match)
Dice = 2 * 3.68M / (4.87M + 3.68M) = 7.36M / 8.55M = 0.86  ✓ High!
```

**Key insight**: Dice loss **does mitigate class imbalance better than pixel-wise accuracy**, but:
- Denominator is **still dominated by majority class** (75% unburned pixels)
- Conservative predictions (favoring unburned) get rewarded
- Model has incentive to be conservative

### Why Dice Helps (vs Accuracy)

| Metric | All-Unburned Prediction | All-Burned Prediction |
|--------|------------------------|-----------------------|
| **Accuracy** | 75% (terrible for burn detection) | 25% (terrible for unburned) |
| **Dice (Unburned)** | 0.86 | 0.0 |
| **Dice (Burned)** | 0.0 | 0.86 |
| **Combined (avg)** | 0.43 | 0.43 |

**Conclusion**: Dice forces attention to *both* classes. A conservative model can't score high on both. But the denominator still weights toward the larger class.

### Practical Implication for Tuning

**Class weighting in loss function (Issue #14, Task 1):**
```python
# Current (implicit weighting by class size)
loss = 0.5 * BCE + 0.5 * Dice

# Proposed: Explicit class weighting
# Burned class: 60% weight, Unburned: 40% weight
burned_weight = 0.6
unburned_weight = 0.4

# Effect: Model penalized MORE for missing burns
# Expected: Recall increases (catches more burns)
# Tradeoff: Precision may decrease (more false positives)
```

**Critical question for Day 4:**
- Does this shift the model from "conservative/high-precision" to "balanced"?
- Or does the data overwhelm the weighting (75% unburned makes it hard regardless)?

---

## Part 2: Pre-Fire vs Post-Fire Only Training

### Current Architecture (Bi-temporal)

**Input shape**: `[B, 2, 12, 512, 512]`
```
[B]: Batch size
[2]: Timesteps (pre-fire, post-fire)
[12]: Spectral bands (Sentinel-2)
[512, 512]: Spatial dimensions
```

**Flattened for model**: `[B, 24, 512, 512]`
- 24 channels = 2 timesteps × 12 bands
- Model learns to **compare pre vs post spectral signatures**
- This is **change detection** (what fundamentally identifies burns)

### Proposed Experiment: Post-Fire Only

**Modified input**: `[B, 1, 12, 512, 512]` → `[B, 12, 512, 512]`

**What changes:**
- Model loses access to pre-fire baseline
- Must recognize **absolute spectral characteristics of burn** (not change)
- Burned vegetation: dark, low NIR reflectance, exposed soil, ash
- Unburned vegetation: green, high NIR reflectance, forest/shrub

**Spectral signatures of burn:**
```
Pre-fire (healthy vegetation):
├─ NIR (Band 7): HIGH (0.4-0.5 reflectance) ← Strong reflectance
├─ Red (Band 3): MEDIUM (0.1-0.2 reflectance)
└─ SWIR (Band 10): MEDIUM (0.1-0.3 reflectance)

Post-fire (burned area):
├─ NIR (Band 7): LOW (0.1-0.2 reflectance) ← Dramatic drop!
├─ Red (Band 3): HIGH (0.2-0.4 reflectance) ← Increased (ash)
└─ SWIR (Band 10): HIGH (0.3-0.5 reflectance) ← Ash signature
```

**Can model recognize burns without change?**
- Theoretically: Yes (absolute signatures are distinctive)
- Practically: Burned areas look like:
  - Dry grassland (low NIR)
  - Rocky areas (low NIR, high SWIR)
  - Water bodies (low NIR, low Red)
  - Shadows (all dark)
- **Ambiguity**: Many non-burn features look "burned"
- **Information loss**: Change baseline removes ambiguity

### Information Cost Analysis

| Aspect | Bi-temporal (Current) | Post-Fire Only (Proposed) |
|--------|----------------------|---------------------------|
| **Discriminative power** | High (change is diagnostic) | Lower (ambiguous signatures) |
| **Model task complexity** | Easier (compare two states) | Harder (classify one state) |
| **Real-world applicability** | Rare (need archived imagery) | Common (single current image) |
| **Expected IoU baseline** | 0.58 | 0.40-0.45 (estimate) |
| **Information loss** | — | ~15-20% performance (estimate) |

### Why Real-World Scenario Matters

**Deployment reality:**
```
Operational burn detection workflow:

Day 1: Fire reported at 14:00
Day 2 (next day): Get satellite imagery
  ├─ Post-fire image: YES (just acquired)
  └─ Pre-fire image: NEED TO ARCHIVE
      ├─ Was archived? Sometimes
      ├─ Correct date range? Often wrong
      └─ Cloud-free? Rare
```

**Current model assumption**: "I always have a clean pre-fire baseline"
**Real-world reality**: "Usually I only have the post-fire image"

**Impact**: Model trained on bi-temporal won't generalize to post-fire-only deployment

---

## Part 3: Dataset Bias vs Real-World Class Imbalance

### CaBuAr Dataset: Artificially Enriched

**Dataset composition:**
- Selected burn scenes from California Burned Areas
- Intentionally enriched: ~25% burned pixels per tile
- Curated to include "interesting" scenes (not random sampling)

**Implication**: Model learns "burns are common" (1 in 4 pixels)

### Real-World Satellite Monitoring: Extreme Sparsity

**Typical deployment scenario:**
```
Region: Central California (10,000 km²)
Sentinel-2 tile: 100 km² = 10,000 pixels × 10,000 pixels = 100M pixels

Burned area per year: 1-3% of region = 100-300 km²
Per tile: 1-3 km² burned per year

Burned pixel density:
  1 km² burned / 100 km² tile = 1% burned pixels
  In best case: 3% burned pixels
```

**Extreme imbalance:**
```
CaBuAr (training): 25% burned, 75% unburned (1:3 ratio)
Real-world (deployment): 1% burned, 99% unburned (1:99 ratio)
                         OR 0.1% burned, 99.9% unburned (1:999 ratio)

Multiplier: Real-world is 8-250× more imbalanced than training!
```

### What This Means for Model Behavior

**Current model (trained on 25% burns):**
```
Validation dataset (also ~25% burns):
- Recall: 60.5% (finds 60% of burns)
- Precision: 94.2% (94% of predictions are correct)

Deployed on real-world (1% burns):
- Recall: ??? (likely similar ~60%)
- Precision: ??? (probably MUCH LOWER)

Why precision drops:
- Current model makes ~17% false positive predictions (1 - 0.942)
- On real data with 1% true burns, FP rate stays similar (~17% of non-burns)
- But non-burns are 99% vs 75% in training
- Result: More FPs in absolute terms, lower precision

Example:
  Training data (25% burns):
    1000 pixels total = 250 burns, 750 non-burns
    Precision 94% means: 94% of "burn" predictions are correct
    If model predicts 300 pixels as burned:
      ├─ Correct: 282 (94%)
      └─ Incorrect: 18 (6%)

  Real-world data (1% burns):
    1M pixels total = 10,000 burns, 990,000 non-burns
    If model makes predictions at SAME RATE (300 "burn" predictions per 1000 pixels):
      ├─ Correct: ~282 (if recall stays same)
      └─ Incorrect: ~18 (from non-burns)
      
      BUT operating on 990k non-burns instead of 750:
      └─ FP rate is still ~2-3 FPs per 1000 non-burns
      └─ Over 990k non-burns: ~2000-3000 FPs
      
      Precision = 282 / (282 + 2000) = 12% (catastrophic drop!)
```

**Key insight**: Precision is NOT calibrated to real-world sparsity.

### Threshold Adjustment Doesn't Fully Solve This

**Naive approach**: "Raise threshold from 0.5 to 0.9 to increase precision"

```
At threshold 0.5 (current): Recall 60%, Precision 94%
At threshold 0.9 (higher): Recall ?, Precision ?

In extreme imbalance (1% burns):
- Higher threshold → fewer predictions → fewer FPs ✓
- But also fewer TPs → recall drops dramatically
- Model must be retrained on actual rare-burn data to be well-calibrated
```

**Better approach**: Retrain on data with actual 1% burn incidence

---

## Part 4: Training/Deployment Mismatch Implications

### The Fundamental Problem

```
┌─────────────────────────────────────────────────┐
│ TRAINING SETUP                                  │
├─────────────────────────────────────────────────┤
│ • Bi-temporal (pre + post fire)                │
│ • 25% class imbalance                          │
│ • Curated scenes (enriched for burns)          │
│ • Metric: IoU on burned class                  │
│ • Result: 0.58 IoU, 94% precision, 60% recall │
└─────────────────────────────────────────────────┘
                    ↓
          Can we deploy this?
                    ↓
┌─────────────────────────────────────────────────┐
│ REAL-WORLD DEPLOYMENT                           │
├─────────────────────────────────────────────────┤
│ • Post-fire only (pre-fire not available)      │
│ • 1% class imbalance (250× sparser)            │
│ • Random imagery (no enrichment)                │
│ • Requirement: High precision (few false alarms)│
│ • Expected: Poor calibration, metric drift     │
└─────────────────────────────────────────────────┘
```

### Where Hyperparameter Tuning Helps (and Doesn't)

**Class weighting helps:**
```
✓ Shifts recall from 60% → 65-70% (better coverage)
✓ Increases F1 score (more balanced)
✓ But: Doesn't change fundamental imbalance mismatch
✗ Real-world still needs to detect 1% class, not 25%
```

**Loss function tuning helps:**
```
✓ Improves IoU by 2-3% on validation set
✓ Better balances precision-recall tradeoff
✓ But: Metric (IoU) is still computed on 25% burn incidence
✗ Real-world IoU will differ significantly at 1% incidence
```

**Extended training helps:**
```
✓ Finds better optimum for current dataset
✓ May improve generalization within similar imbalance
✓ But: Doesn't close 8-250× imbalance gap to real-world
✗ Model is still "optimized for 25% burns, not 1% burns"
```

### What Would Actually Help

1. **Synthetic rare-burn dataset** (high effort)
   - Resample CaBuAr to 1% burn incidence
   - Retrain model on actual deployment distribution
   - Recalibrate metrics

2. **Post-fire-only variant** (medium effort)
   - Train model without pre-fire imagery
   - Tests if absolute signatures sufficient (informative regardless)
   - More realistic for operational deployment

3. **Threshold optimization on real data** (medium effort)
   - Deploy current model
   - Collect real-world predictions with ground truth
   - Find optimal threshold for real imbalance
   - May require threshold adjustment per region

4. **Production monitoring** (ongoing)
   - Track precision/recall on deployed model
   - Alert if metrics degrade (model drift)
   - Retrain annually with new labeled burns

---

## Part 5: Day 4 Sprint Implications

### What Hyperparameter Tuning CAN Achieve

**Realistic goals for Issues #14-15:**
- [ ] Improve recall: 60% → 65-70% (find more burns in training distribution)
- [ ] Improve IoU: 0.584 → 0.61-0.63 (better boundary detection)
- [ ] Better F1: 0.737 → 0.75+ (more balanced)
- [ ] **Applies to**: 25% burn incidence (similar to CaBuAr)

### What Hyperparameter Tuning CANNOT Achieve

**Impossible without retraining:**
- [ ] Calibrate to 1% burn incidence
- [ ] Make post-fire-only variant work well
- [ ] Reduce false positive rate at extreme imbalance
- [ ] Guarantee real-world deployment performance

### Recommended Day 4 Approach

**Primary focus** (Issues #14-15):
- Class weighting + loss tuning for **maximum improvement within CaBuAr distribution**
- This is legitimate: Better model on training distribution is useful
- Document clearly: "Improvements valid for 20-30% burn prevalence"

**Secondary investigation** (Optional, low cost):
- Quick experiment: Drop pre-fire imagery (20 min GPU)
- Measure information loss from bi-temporal → post-fire only
- Informs production architecture choice
- Result: "Information loss = ~15-20% IoU" (or measured value)

**Documentation** (This document + Day 4 results):
- Flag the training/deployment mismatch explicitly
- Recommend: Future work should include rare-burn dataset
- Suggest: Operational deployment needs offline calibration on real data

---

## Part 6: Experimental Recommendations

### If Time Permits in Day 4 (Optional Experiment)

**Experiment: Post-Fire Only Variant**

```
Goal: Quantify information loss from dropping pre-fire

Setup:
  - Duplicate 03_training.ipynb → 03_training_postfire_only.ipynb
  - Modify data loading to use only post-fire images
  - Reduce model input channels: 24 → 12
  - Keep all else same (class weighting, loss, epochs)

Training:
  - Run 20 epochs
  - Track: Best validation IoU, final recall, final precision

Evaluation:
  - Run full test set inference
  - Compare metrics to bi-temporal baseline

Cost: ~0.33 GPU hours (20 min training + 10 min eval)

Decision Rule:
  IF IoU drops 10-20% (0.58 → 0.48-0.52):
    THEN pre-fire is valuable but not critical
    RECOMMENDATION: Consider post-fire-only for deployment
    
  ELSE IF IoU drops 20%+ (0.58 → <0.48):
    THEN change detection is critical
    RECOMMENDATION: Always archive pre-fire for production
    
  ELSE IF IoU drops <10% (0.58 → 0.52+):
    THEN absolute signatures sufficient
    RECOMMENDATION: Post-fire-only model is viable
```

### If Full Investigation Planned (Future Sprint)

**Issue #24 (proposed): Real-World Deployment Robustness**

```
Components:

1. Synthetic rare-burn dataset
   - Resample CaBuAr to 1% burn incidence
   - Train variant on actual real-world distribution
   
2. Real-world calibration study
   - Deploy current model operationally
   - Collect ground truth labels
   - Measure actual precision/recall at 1% incidence
   - Compare to validation metrics
   
3. Production deployment guide
   - Document threshold adjustment procedures
   - Recommend monitoring/retraining schedule
   - Flag when model accuracy degrades
```

---

## Part 7: Summary & Recommendations

### Key Findings

1. **Dice loss mitigates (but doesn't eliminate) class imbalance**
   - Denominator still dominated by majority class
   - Class weighting is appropriate intervention

2. **Pre-fire imagery is valuable but creates deployment mismatch**
   - Information cost if dropped: ~15-20% IoU (estimated)
   - Real deployment often lacks clean pre-fire reference

3. **Training/deployment class imbalance mismatch is severe**
   - CaBuAr: 25% burns
   - Real-world: 0.1-1% burns
   - Precision metrics likely invalid for real deployment

4. **Hyperparameter tuning improves within training distribution**
   - But doesn't close deployment gap
   - Realistic target: +5-10% recall, +2-5% IoU
   - This is good and worth doing
   - But label what it is: "Optimized for 25% burn incidence"

### Day 4 Sprint Recommendations

**Do** (Issues #14-15):
- ✓ Implement class weighting (high ROI, low cost)
- ✓ Experiment with loss ratios (good exploration, medium cost)
- ✓ Document all results clearly
- ✓ Label improvements as "within 25% burn training distribution"

**Consider** (Optional):
- ◐ Post-fire-only experiment (low cost, high insight)
- ◐ Document information loss from dropping pre-fire
- ◐ Flag deployment calibration needs

**Defer** (Future sprints):
- ⊘ Rare-burn dataset retraining (expensive, important)
- ⊘ Real-world operational evaluation (requires deployment)
- ⊘ Production monitoring system (post-deployment)

### Critical Flag for Documentation

**After Day 4, update BASELINE_ANALYSIS.md:**

```markdown
## Important: Training/Deployment Mismatch

This model is trained and evaluated on CaBuAr dataset with ~25% burn prevalence.
Real-world burn detection will encounter 1% burn prevalence (or lower).

**Implications for deployment:**
- Metrics (IoU, precision, recall) are valid for 20-30% burn scenes
- For 1% burn scenes, precision will be lower than reported
- Model may require recalibration on operational data
- Recommend: Collect ground truth from actual deployment
            Retrain or adjust thresholds based on real performance

**Recommendations:**
- Keep pre-fire imagery when available (improves IoU ~15-20%)
- Use threshold 0.35-0.40 for operational deployment (balances precision/recall)
- Monitor false positive rate in production
- Plan annual retraining with new labeled burns
```

---

## References

- Baseline Model: docs/BASELINE_RESULTS.md
- Test Set Analysis: docs/BASELINE_ANALYSIS.md
- Class Imbalance in Deep Learning: [Review articles on imbalanced segmentation]
- Dice Loss Properties: [TBD - technical references]
- Real-World Burn Detection: CaBuAr dataset paper, Copernicus Emergency Management

---

**Document Purpose**: Investigative exploration to inform Day 4 hyperparameter tuning strategy  
**Next Action**: Reference during Day 4 execution; incorporate findings into final documentation  
**Related Issues**: #13 (Day 4 Sprint), #14 (Hyperparameter Tuning), #15 (Loss Optimization), #20 (Final Documentation)

---

**Author Note**: This investigation emerged from asking "why" about fundamental assumptions in the baseline model. The insights expose gaps between training and deployment that hyperparameter tuning alone cannot close, but should inform long-term model development strategy.
