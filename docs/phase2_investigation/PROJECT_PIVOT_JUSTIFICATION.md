# Project Pivot: From Hyperparameter Tuning to Spectral Relabeling

**Date**: 2026-06-24  
**Reason**: Critical data quality discovery during inference analysis  
**Status**: Pivot approved, moving to Phase 3

---

## Summary

After completing Day 4 hyperparameter tuning (Issue #14) and analyzing test set predictions, we discovered that **the ground truth labels are fundamentally ambiguous**. They conflate recent burn scars with old recovered burn scars from previous fires.

**This makes the original project direction (optimizing a binary classifier) untenable.**

**New direction**: Build a spectral labeling system to separate burn age/severity classes, then train an RGB+IR model for cross-sensor transfer to NAIP.

---

## The Evidence: Tile 18 Analysis

### Visual Proof

See: `docs/phase2_investigation/analysis_artifacts/tile18_prepost_difference_analysis.png`

**Tile 18 Characteristics**:
- Ground truth label: **193,864 burned pixels (74% of tile)**
- Pre-fire image: Healthy vegetation mosaic (false color: SWIR/NIR/Red)
- Post-fire image: **Nearly identical to pre-fire**
- Model prediction: Only 14% burned (mostly green areas predicted as unburned)

### Spectral Analysis Results

| Metric | Finding | Implication |
|--------|---------|-------------|
| **NDVI change** | Minimal, scattered | No dramatic vegetation loss (not recent fire) |
| **SWIR band change** | Predominantly DECREASES | Vegetation getting greener, not burned charcoal |
| **Change magnitude** | Subtle, patchy | Consistent with vegetation recovery, not fire event |
| **Change distribution** | Random scatter | Does NOT align with "burned" mask |

### The Contradiction

```
Ground Truth Says:  "74% of this tile is burned"
Spectral Data Says: "This tile shows vegetation recovery, not recent fire"
Model Predicts:     "Only 14% burned" ← Actually correct!
```

---

## Root Cause: Mislabeled Data

**What the ground truth actually contains:**

1. **Recent fires** (2-4 weeks old): Strong spectral change, clear in difference maps
2. **Old burn scars** (6+ months): Recovered vegetation, minimal spectral change
3. **Recovering areas** (2-6 months): Intermediate spectral signatures
4. **No distinction** in binary labels between recent and old

**Why this breaks the model:**

The model is trained to detect **spectral change** (how images differ between pre and post-fire). But:
- Recent burns: Dramatic change ✓ Model learns this well
- Old recovered burns: Minimal change ✗ Model correctly outputs "no change" (unburned)
- Ground truth labels both as "burned" ✗ Contradiction

**The high variance in per-sample performance makes sense now:**

- Tiles with recent fires: Model performs well (65-91% recall)
- Tiles with old recovered scars: Model performs poorly (0-15% recall)
- Model isn't broken; data is ambiguous

---

## Why Hyperparameter Tuning Can't Fix This

**What we tried:**
- pos_weight=1.5 to penalize false negatives
- Better loss function combinations
- Threshold adjustment

**What happened:**
- Aggregate recall improved (60% → 71%) ✓
- But per-sample variance stayed high (std dev 34.4%) ✗
- Model still fails on large tiles with old scars ✗

**Why it can't work:**
You can't train a spectral change detector to detect things that don't show spectral change. The problem isn't the model or hyperparameters—it's the data.

---

## The Solution: Spectral Relabeling

**New approach:**

1. **Analyze all 12 Sentinel-2 bands** to identify burn age/severity
   - Recent burn: Strong NDVI drop, high NBR change, spectral consistency
   - Old scar: Partial recovery, intermediate NDVI, patchy signature
   - Recovering: Mixed signals, gradual vegetation return
   - Healthy: Stable, high NDVI, no change

2. **Create multi-class labels** instead of binary
   - This removes ambiguity and gives model clear patterns to learn

3. **Train on RGB+IR only** (4 channels)
   - Matches NAIP capability for transfer learning

4. **Test zero-shot transfer** to NAIP
   - Validates spectral principles transcend sensors

---

## Impact on Project Timeline

| Phase | Original Plan | New Plan | Status |
|-------|---|---|---|
| **Phase 1** | Baseline model | ✓ Baseline complete | Complete |
| **Phase 2** | Hyperparameter tuning | ✓ Investigation + discovery | Complete |
| **Phase 3** | Loss optimization | **→ Spectral relabeling** | Starting |
| **Phase 4** | Extended training | **→ Multi-class training** | Planned |
| **Phase 5** | — | **→ Cross-sensor transfer** | Planned |

---

## Learning Outcomes Alignment

This pivot better serves PA3 learning outcomes:

✓ **Data quality assessment**: Recognizing and addressing labeling issues  
✓ **Spectral analysis**: Using multiple bands to make quantitative decisions  
✓ **Multi-class classification**: Going beyond binary to meaningful categories  
✓ **Transfer learning**: Testing generalization across sensors  
✓ **Scientific rigor**: Investigating why the model "fails" → discovering the real problem  

---

## Conclusion

**The original direction was pursuing better hyperparameters for a fundamentally flawed task.** This discovery that **the model is actually working correctly on ambiguous data** is more valuable than any hyperparameter tuning result.

By pivoting to spectral relabeling and multi-class training, we address the root cause and build toward a more robust cross-sensor transfer learning system.

**Recommendation**: Proceed with Phase 3 (Spectral Relabeling Strategy).

---

**Next Document**: `docs/phase3_relabeling/SPECTRAL_RELABELING_STRATEGY.md`
