# Phase II_02: Change-Detection Architecture Decision

**Date**: 2026-06-24  
**Status**: Strategy Locked  
**Owner**: RETINNA Investigation

---

## Decision: Difference-Based Change Detection Input

### What We're Doing
Train U-Net on **spectral change images**, not absolute reflectance:

```
Input:  (Post_RGBN - Pre_RGBN) = 4 channels
        - (Post_R - Pre_R)
        - (Post_G - Pre_G)
        - (Post_B - Pre_B)
        - (Post_NIR - Pre_NIR)

Output: 7-class severity label from Phase II_01
        (Unburned, Low, Moderate, High, Extreme, Water, Cloud)
```

### Why This Approach

#### 1. **Aligns with Phase II_01 Labels** ✓
- Phase II_01 labels are change-based (RdNBR = dNBR / sqrt(|NBRpre|))
- Input being change-based (Post - Pre) is coherent with change-based labels
- Model learns: "this change magnitude = this severity class"

#### 2. **Terrain Normalization** ✓
- Sparse chaparral, grassland, forest have different "normal" reflectance
- Difference removes terrain effects
- Model learns burn patterns relative to local baseline (pre-fire state)
- Same fire signature works across different vegetation types

#### 3. **Phase III Transfer Ready** ✓
- NAIP has 4 bands (RGBN) - exactly what we're using
- Can compute (Post_NAIP - Pre_NAIP) directly in Phase III
- No need to relearn from S2; model transfers directly
- No feature extraction or band adaptation needed

#### 4. **Robust to Seasonal/Regional Variation** ✓
- Trained on relative change, not absolute values
- Works whether pre-fire is sparse or lush
- Works whether pre-fire has snow, standing water, or shadows
- Generalizes better across geographies

#### 5. **Professional Standard** ✓
- Change detection is how remote sensing does temporal analysis
- dNBR, NDVI-change, spectral indices are all difference-based
- Proven approach in published literature

### Why NOT Post-Only (8 channels)

If we used `[Pre_RGBN + Post_RGBN]` (8 channels):
- ✅ Simpler: uses absolute spectral values
- ✅ More information: model has both pre and post
- ❌ Doesn't transfer to Phase III without modification
- ❌ Model learns absolute values (chaparral vs forest), not change
- ❌ Weaker terrain normalization

### Coherence with Phase II_01

Phase II_01 computed RdNBR:
- dNBR = NBRpre - NBRpost (change in vegetation index)
- RdNBR = dNBR / sqrt(|NBRpre|) (normalized by pre-fire amount)
- USGS thresholds applied to RdNBR

Phase II_02 input:
- (Post_RGBN - Pre_RGBN) (change in reflectance)
- Both measure change, just different metrics
- Labels capture severity of change, input captures change signal
- No conflict; perfect alignment

---

## Implementation Details

### Data Preparation (Phase II_02 Notebook)

1. Load Phase II_01 labels (post-fire severity)
2. Extract CaBuAr S2 data: bands [2,3,4,8] = RGBN
3. For each sample:
   ```python
   pre_rgbn = S2[sample_idx, :, :, [2,3,4,8]]  # Pre-fire, 4 channels
   post_rgbn = S2[sample_idx, :, :, [2,3,4,8]]  # Post-fire, 4 channels
   diff_input = post_rgbn - pre_rgbn  # 4 channels
   label = phase_ii_01_labels[sample_idx]  # From RdNBR classification
   ```

4. Stack: `input_tensor = diff_input` (4 channels, 512×512)

### Model Architecture

```python
class BurnSeverityUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNet(in_channels=4, out_channels=7)
        # Input: 4 channels (RGBN difference)
        # Output: 7 classes (severity)
```

### Training Strategy

- **Input channels**: 4 (NAIP-compatible)
- **Output classes**: 7 (severity from Phase II_01)
- **Training samples**: 424 (each pre/post pair → 1 sample)
- **Train/Val/Test split**: Use fold assignment from metadata
- **Loss**: Weighted cross-entropy (class imbalance from Phase II_01)
- **Metrics**: Per-class IoU, pixel accuracy, confusion matrix

### Phase III Transfer

When moving to NAIP:
```python
# Phase III: NAIP pre/post pairs
naip_pre = load_naip_tile(tile_id, date_pre, bands=['R','G','B','NIR'])
naip_post = load_naip_tile(tile_id, date_post, bands=['R','G','B','NIR'])
naip_diff = naip_post - naip_pre  # 4 channels

# Inference
prediction = trained_model(naip_diff)  # Direct, no adaptation needed
```

---

## Trade-Offs Accepted

| Aspect | Trade-Off |
|--------|-----------|
| **Input information** | Less absolute data, more relevant change signal |
| **Training flexibility** | Requires pre/post pairs, can't use unpaired images |
| **Atmospheric noise** | Difference amplifies atmospheric variation between dates |
| **Model interpretability** | Harder to debug; focus on relative patterns not absolute values |

All acceptable because change-detection aligns with Phase III requirements.

---

## Files Affected

- **Phase II_02 notebook**: New implementation with difference-based input
- **Phase II_MASTER.md**: Updated with this decision
- **Phase II_INDEX.md**: Points to this strategy doc

---

## Next Steps

1. ✅ Decision locked
2. ⏳ Create II_02_unet_training.ipynb with difference-based input
3. ⏳ Train on 424 samples (pre/post pairs)
4. ⏳ Validate on 78 + 68 = 146 samples
5. ⏳ Test Phase III transfer strategy

---

**Status**: Ready to implement  
**Implementation file**: [II_02_unet_training.ipynb](../notebooks/II_02_unet_training.ipynb)
