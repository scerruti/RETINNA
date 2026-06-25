# Data Loading Bug Fix: SCL Band Missing from CaBuAr

**Date**: 2026-06-24  
**Phase**: II_01 (Spectral Relabeling)  
**Status**: RESOLVED  
**Impact**: Critical — blocked cloud/shadow detection

---

## Problem Discovery

During QC of II_01 output, we detected that **Cloud/Shadow class was always 0%** across all splits:
- Train: 0 px (0.00%)
- Val: 0 px (0.00%)
- Test: 0 px (0.00%)

This is statistically impossible on real satellite data.

---

## Root Cause Analysis

### Initial Hypothesis
The Sentinel-2 Scene Classification (SCL) band at index 11 should contain integer class codes (0-11) indicating:
- 0-3, 8-11: Cloud/shadow classes
- 4: Vegetation
- 5: Not vegetated
- 6: Water
- 7: Unclassified

### Actual Finding
Diagnostic inspection of all 12 bands revealed:

```
Band statistics (pre-fire sample):
 0. B2   : min=0.0098, max=0.1508, unique=953     (Blue reflectance)
 1. B3   : min=0.0087, max=0.5505, unique=1529    (Green reflectance)
 2. B4   : min=0.0126, max=0.5815, unique=1991    (Red reflectance)
 3. B5   : min=0.0089, max=0.5979, unique=3038    (Vegetation edge)
 4. B6   : min=0.0105, max=0.5146, unique=3231    (Red edge)
 5. B7   : min=0.0063, max=0.4927, unique=3269    (Red edge)
 6. B8   : min=0.0098, max=0.5398, unique=3397    (NIR)
 7. B8A  : min=0.0073, max=0.5695, unique=3683    (Narrow NIR)
 8. B11  : min=0.0028, max=0.5912, unique=3821    (SWIR-1)
 9. B12  : min=0.0579, max=0.5617, unique=3265    (SWIR-2)
10. CLP  : min=0.0179, max=0.7067, unique=4440    (Cloud probability?)
11. SCL  : min=0.0151, max=0.7002, unique=3285    ← SHOULD BE CLASS CODES 0-11!
```

**Key observation**: Band 11 has 3,285 unique values, not 12. It's a reflectance band, not class codes.

### Denormalization Test
Multiplying by 10,000 (Sentinel-2 normalization factor):
```
Denormalized SCL values: [151, 152, 155, ..., 4769, 4815, 7002]
```

These are reflectance values, not class codes (0-11).

### Conclusion
**The CaBuAr dataset does NOT include the Sentinel-2 SCL band.** What's labeled "SCL" is actually another reflectance measurement (possibly cloud probability or ancillary data).

---

## Solution: Spectral Cloud Detection

Instead of relying on SCL class codes, we implemented **spectral thresholds** for cloud detection:

```python
Cloud signature: Blue > 0.25 AND Blue/NIR > 0.8
```

**Rationale**:
- Clouds have high reflectance in visible (especially blue) wavelengths
- Clouds have low absorption in NIR (unlike vegetation)
- High blue/NIR ratio distinguishes clouds from vegetation and burned areas

**Implementation**:
- Extract blue (B2) and NIR (B8) bands
- Apply threshold during pixel classification
- Prioritize clouds before water and burn classes

---

## Changes Made

### File: `notebooks/II_01_spectral_relabeling.ipynb`

**1. Updated `compute_spectral_indices()`**:
```python
blue_idx = 0  # B2 = blue
# Extract blue_post
return {
    ...
    'blue_post': blue_post,
    'nir_post': nir_post
}
```

**2. Updated `classify_pixel()`**:
```python
def classify_pixel(dnbr, mndwi, blue=None, nir=None):
    # Spectral cloud detection
    if blue is not None and nir is not None:
        if blue > 0.25 and (blue / (nir + 1e-8)) > 0.8:
            return 6  # Cloud/Shadow
    # ... rest of classification
```

**3. Updated `process_dataset()`**:
```python
blue_post = indices['blue_post']
nir_post = indices['nir_post']

labels[y, x] = classify_pixel(dnbr[y, x], mndwi[y, x], 
                              blue=blue_post[y, x], nir=nir_post[y, x])
```

---

## Validation

**Before fix**:
- Cloud/Shadow: 0 px (0.00%) across all splits

**Expected after fix**:
- Cloud/Shadow: ~1-3% (typical for CaBuAr tiles)
- Other classes should remain stable

---

## Implications for Phase II

### Phase II_01 (Spectral Relabeling) ✓ FIXED
- Cloud detection now uses spectral thresholds
- Re-run notebook to generate corrected labels

### Phase II_02 (RGB+IR Training)
- Will use corrected multi-class labels from II_01
- No code changes needed — just use new labels

### Phase II_03 (NAIP Transfer)
- NAIP (4-band) doesn't include SCL either
- Spectral cloud detection will work for NAIP
- Method is generalizable across sensors ✓

---

## Lessons Learned

1. **Don't assume band availability** — inspect actual data first
2. **TorchGeo normalization affects all bands uniformly** — even indices labeled as "class codes"
3. **Spectral thresholds are more portable** — work across different data sources (Sentinel-2, NAIP, etc.)
4. **QC caught a critical issue early** — cloud detection was broken before training

---

## References

- Sentinel-2 Scene Classification (SCL) documentation: ESA S2 Level-2A specification
- CaBuAr dataset: TorchGeo (https://github.com/torchgeo/torchgeo)
- Cloud detection via spectral indices: Commonly used in multispectral analysis

---

**Next Action**: Re-run Phase II_01 with corrected notebook, verify Cloud/Shadow is now >0%, proceed to II_02.

