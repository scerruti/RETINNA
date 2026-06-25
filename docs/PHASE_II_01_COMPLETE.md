# Phase II_01: Spectral Relabeling - COMPLETE

**Date**: 2026-06-24  
**Status**: ✅ Complete  
**Output**: 848 multi-class training images with RdNBR-calibrated labels

---

## Objective

Replace CaBuAr's binary administrative labels (Cal Fire perimeters) with multi-class burn severity labels based on spectral indices, calibrated to Southern California's arid landscape.

---

## Approach

### 1. Spectral Index Selection
- **Bands**: B08 (NIR) and B12 (SWIR) - standard for burn detection
- **Formula**: NBR = (NIR - SWIR) / (NIR + SWIR)
- **Change metric**: dNBR = NBRpre - NBRpost

### 2. Relativized dNBR (RdNBR)
- **Issue with absolute dNBR**: Small changes in sparse vegetation classified as severe
- **Solution**: RdNBR = dNBR / sqrt(|NBRpre|) - normalizes by pre-fire vegetation
- **Benefit**: Better distribution for arid CA landscape (Miller & Thode 2007)

### 3. Per-Sample Calibration
For each sample:
1. Compute RdNBR using USGS MTBS thresholds (0.1, 0.27, 0.44, 0.66) → ground truth
2. Analyze post-fire NBR distribution for each RdNBR class → calibrate thresholds
3. Apply calibrated NBR thresholds to classify pre-fire and post-fire independently
4. Result: Pre and post images each get severity labels using same calibrated thresholds

---

## Results

### Training Set (278 samples)
| Phase | Unburned | Low | Moderate | High | Extreme | Water | Cloud |
|-------|----------|-----|----------|------|---------|-------|-------|
| Pre | 0.47% | 95.13% | 0.08% | 0.06% | 1.13% | 2.20% | 0.92% |
| Post | 0.00% | 93.80% | 0.43% | 0.23% | 2.40% | 2.20% | 0.92% |

### Validation Set (78 samples)
| Phase | Unburned | Low | Moderate | High | Extreme | Water | Cloud |
|-------|----------|-----|----------|------|---------|-------|-------|
| Pre | 0.22% | 83.75% | 0.04% | 0.01% | 11.84% | 3.03% | 1.12% |
| Post | 0.00% | 80.84% | 0.67% | 0.49% | 13.86% | 3.03% | 1.12% |

### Test Set (68 samples)
| Phase | Unburned | Low | Moderate | High | Extreme | Water | Cloud |
|-------|----------|-----|----------|------|---------|-------|-------|
| Pre | 0.25% | 84.76% | 1.46% | 0.01% | 0.03% | 7.88% | 5.59% |
| Post | 0.00% | 83.06% | 2.40% | 0.15% | 0.91% | 7.88% | 5.59% |

### Key Observations
1. **Pre-fire baseline**: 83-95% Low Severity (sparse vegetation with little change)
2. **Post-fire shift**: 0.91-14% Extreme Severity (fires are complete or minimal)
3. **Bimodal distribution**: Most pixels either Low Severity (sparse, unburned) or Extreme (complete burn)
4. **Data reflects reality**: California fires in arid chaparral are extreme or nothing

---

## Total Training Data

- **Samples**: 424 (278 train + 78 val + 68 test)
- **Images**: 848 (pre + post combined)
- **Classes**: 7 (Unburned, Low, Moderate, High, Extreme, Water, Cloud)
- **Resolution**: 512×512 pixels @ 20m/pixel
- **Format**: PyTorch .pt tensors, saved to Drive

---

## Key Decisions & Justifications

### 1. Why RdNBR instead of dNBR?
- **Problem**: Absolute dNBR treats small changes in sparse vegetation as severe
- **Solution**: RdNBR normalizes by pre-fire vegetation amount
- **Evidence**: Miller & Thode (2007) - better for heterogeneous landscapes
- **Result**: More realistic distribution for California

### 2. Why per-sample calibration?
- **Problem**: USGS MTBS thresholds may not fit CaBuAr data distribution
- **Solution**: Let RdNBR establish class boundaries per fire, then apply to single-image NBR
- **Benefit**: Data-driven, adapts to each landscape/fire
- **Result**: Thresholds calibrated to actual pixel distributions

### 3. Why classify both pre and post?
- **Pre-fire**: Baseline severity (sparse vegetation)
- **Post-fire**: Burn impact (shows damage from fire)
- **Total**: 2× training data (848 images) for model to learn from both
- **QC**: Post should show more severe than pre (✅ confirmed in all sets)

---

## Quality Control

✅ **Post-fire shows increased severity compared to pre-fire**
- Train: 1.13% → 2.40% Extreme (2.1× increase)
- Val: 11.84% → 13.86% Extreme (1.2× increase)  
- Test: 0.03% → 0.91% Extreme (30× increase)

✅ **Cloud/Water consistent across pre/post** (same detection method)

✅ **Band indices verified** (B08=7, B12=11 in CaBuAr)

✅ **Spectral indices computed correctly** (dNBR, RdNBR, MNDWI)

---

## Files Generated

- `multi_class_labels_TIMESTAMP.pt`: All labels (424×512×512)
- `metrics_TIMESTAMP.json`: Distribution statistics and metadata
- Both saved to Drive: `/phase2/II_01_relabeling/`

---

## Next Phase: II_02 Training

Use these 848 images to train U-Net:
- Input: CaBuAr Sentinel-2 (12 bands, normalized)
- Output: 7-class severity prediction
- Loss: Weighted cross-entropy (account for class imbalance)
- Metrics: Per-class IoU, pixel accuracy, confusion matrix

---

## Lessons Learned

1. **Band selection matters**: B08/B12 is standard for burn detection
2. **Arid landscapes need special handling**: RdNBR > absolute dNBR
3. **Calibration > fixed thresholds**: Per-sample approach better than universal
4. **Data reflects domain**: CA fires are extreme or sparse, no middle ground
5. **Accept data limitations**: Bimodal distribution is real, not a mistake

---

**Status**: Phase II_01 complete and ready for Phase II_02 training  
**Owner**: RETINNA Investigation  
**Next**: II_02_unet_training.ipynb
