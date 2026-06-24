# Phase 3: Spectral Relabeling Strategy

**Objective**: Create multi-class burn labels using official USGS burn severity classifications  
**Input**: CaBuAr dataset with original binary labels (burned/unburned)  
**Output**: Multi-class labels based on USGS Monitoring Trends in Burn Severity (MTBS) standard  
**Purpose**: Enable training of robust RGB+IR model for cross-sensor transfer to NAIP  
**Authority**: USGS MTBS program, Forest Service burn severity standards

---

## Problem Statement

**Current limitation of binary labels**:
```
Ground Truth: "Burned" (1) or "Unburned" (0)
Reality:      Unburned | Low severity | Moderate | High severity | Extreme + Water + Clouds
Problem:      Old scars and recent fires both labeled "burned" → model can't distinguish
```

**Example from Tile 18**:
- 74% labeled as "burned"
- But spectral data shows vegetation recovery (SWIR decreases, NDVI stable)
- This is old scar recovery or moderate severity, not high severity fire
- Model correctly predicts "unburned" → marked as wrong

**Solution**: Replace binary labels with official USGS burn severity classes that reflect spectral reality and established standards.

---

## Proposed Multi-Class Scheme

### Foundation: USGS Burn Severity Standards

Classes are based on **Normalized Burn Ratio (NBR)** thresholds from:
- **USGS Monitoring Trends in Burn Severity (MTBS) Program**
- **US Forest Service Burn Severity Classification**
- **Key et al., 2006** and **USGS Remote Sensing Phenology Lab**

NBR is computed as:
```
NBR = (NIR - SWIR) / (NIR + SWIR)
dNBR = NBRpre - NBRpost  (change)
```

---

### Class Definitions

#### Class 0: Unburned/Healthy
**USGS Definition**: Unburned or no significant change  
**Threshold**: `NBR > 0.1` OR `dNBR < 0.05`  
**Spectral indicators**:
- High NDVI (> 0.5)
- Stable or increasing NIR
- No major spectral change across bands
- Water index (NDWI) < 0.3

**Interpretation**: No fire impact or natural vegetation stability

**Source**: USGS MTBS - Unburned class

---

#### Class 1: Low Severity
**USGS Definition**: Low burn severity, light surface scorch  
**Threshold**: `0.05 < NBR <= 0.1` OR `0.05 < dNBR <= 0.27`  
**Spectral indicators**:
- Moderate NDVI decrease (0.3-0.5)
- Partial NIR retention
- Some vegetation still present
- Red band shows minor increase

**Interpretation**: Light burn with vegetation survival, quick recovery expected

**Source**: USGS MTBS - Low Severity class

---

#### Class 2: Moderate Severity
**USGS Definition**: Moderate burn severity, more substantial cover change  
**Threshold**: `-0.1 < NBR <= 0.05` OR `0.27 < dNBR <= 0.44`  
**Spectral indicators**:
- Significant NDVI decrease (0.0-0.3)
- Marked NIR decrease
- Exposed soil visible in some areas
- Mix of burned and unburned patches

**Interpretation**: Visible burn scar, substantial vegetation loss, multi-year recovery

**Source**: USGS MTBS - Moderate Severity class

---

#### Class 3: High Severity
**USGS Definition**: High burn severity, large areas of complete consumption  
**Threshold**: `-0.27 < NBR <= -0.1` OR `dNBR > 0.44`  
**Spectral indicators**:
- Large NDVI decrease (< 0.0, often negative)
- Very low NIR (charred/bare)
- Dominant red/SWIR signal (exposed soil/charred material)
- Coherent fire signature across bands

**Interpretation**: Complete vegetation removal, severe burn damage, long recovery (10+ years)

**Source**: USGS MTBS - High Severity class

---

#### Class 4: Extreme Severity
**USGS Definition**: Extreme burn severity, complete consumption with soil/terrain change  
**Threshold**: `NBR < -0.27`  
**Spectral indicators**:
- Very low/negative NDVI
- Minimal NIR signal
- Strong SWIR signal (exposed charred material)
- Potential for terrain change (erosion)

**Interpretation**: Complete ecosystem conversion, potential mudslide risk, extreme long-term impacts

**Source**: USGS MTBS - Extreme Severity class

---

#### Class 5: Water/Wetlands
**Definition**: Water bodies, permanent or semi-permanent  
**Threshold**: `MNDWI > 0.3` OR `NDVI < -0.2`  
**Spectral indicators**:
- Modified Normalized Difference Water Index (MNDWI) high
- Strong absorption in NIR and SWIR
- Very low NDVI
- Consistent across time

**Interpretation**: Water body, lakes, rivers, wetlands, reservoirs

**Source**: McFEE et al. (2013), Sentinel-2 water detection standards

---

#### Class 6: Cloud/Shadow/Invalid
**Definition**: Clouds, shadows, sensor artifacts, no-data areas  
**Threshold**: `SCL band == {0,1,2,3,8,9,10,11}` (Sentinel-2 classification)  
**Spectral indicators**:
- Sentinel-2 Scene Classification Layer (SCL) flags cloud/shadow
- Extreme/impossible values
- No valid data

**Interpretation**: Exclude from analysis (noisy/unreliable)

**Source**: ESA Sentinel-2 Level-2A SCL documentation

---

## Implementation: Labeling Algorithm

### Stage 1: Compute Spectral Change Metrics

For each pixel in each tile:

```python
# Load pre and post-fire images
pre_fire = image[0, :, :, :]   # [12, 512, 512]
post_fire = image[1, :, :, :]  # [12, 512, 512]

# Extract key bands
# Sentinel-2 order: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, CLP, SCL
red_idx, nir_idx, swir1_idx, swir2_idx = 2, 6, 10, 11

red_pre = pre_fire[red_idx]
red_post = post_fire[red_idx]
nir_pre = pre_fire[nir_idx]
nir_post = post_fire[nir_idx]
swir_pre = pre_fire[swir1_idx]
swir_post = post_fire[swir1_idx]

# Compute spectral indices
ndvi_pre = (nir_pre - red_pre) / (nir_pre + red_pre + eps)
ndvi_post = (nir_post - red_post) / (nir_post + red_post + eps)
ndvi_change = ndvi_pre - ndvi_post

nbr_pre = (nir_pre - swir_pre) / (nir_pre + swir_pre + eps)
nbr_post = (nir_post - swir_post) / (nir_post + swir_post + eps)
nbr_change = nbr_pre - nbr_post

# Individual band changes
delta_red = red_post - red_pre
delta_nir = nir_post - nir_pre
delta_swir = swir_post - swir_pre

# Magnitude of change
change_magnitude = sqrt(delta_red^2 + delta_nir^2 + delta_swir^2)

# Coherence: Do changes agree across multiple indices?
# (high coherence = fire, low coherence = noise/mixed signals)
coherence_score = correlation(ndvi_change, nbr_change, delta_red, delta_nir)
```

### Stage 2: Classification Rules

Apply official USGS thresholds to assign class:

```python
# Pre-compute indices
nbr_pre = (nir_pre - swir_pre) / (nir_pre + swir_pre + eps)
nbr_post = (nir_post - swir_post) / (nir_post + swir_post + eps)
dnbr = nbr_pre - nbr_post  # dNBR = pre - post (positive = burned)

mndwi = (green - swir) / (green + swir + eps)  # Modified NDWI for water
scl = sample['scl_band']  # Sentinel-2 Scene Classification Layer

# Classification (in priority order)
if scl in [0, 1, 2, 3, 8, 9, 10, 11]:  # Cloud/Shadow/Invalid
    class = 6  # Cloud/Shadow
    
elif mndwi > 0.3:  # Water detection
    class = 5  # Water/Wetlands
    
elif dnbr < -0.27:  # Extreme severity
    class = 4  # Extreme Severity (USGS)
    
elif -0.27 <= dnbr <= -0.1:  # High severity
    class = 3  # High Severity (USGS)
    
elif -0.1 < dnbr <= 0.05:  # Moderate severity
    class = 2  # Moderate Severity (USGS)
    
elif 0.05 < dnbr <= 0.27:  # Low severity
    class = 1  # Low Severity (USGS)
    
else:  # dnbr >= 0.27 or minimal change
    class = 0  # Unburned (USGS)
```

**Authority**: All thresholds from USGS MTBS (Monitoring Trends in Burn Severity)

### Stage 3: Quality Control

**Flag and investigate outliers**:
- Pixels with extreme values → likely cloud/shadow
- Pixels with contradictory indices (high NDVI change but low NBR) → ambiguous
- Regions with sparse valid data → exclude
- Check agreement with original binary labels

**Validation approach**:
1. Manually inspect samples from each class
2. Verify spectral patterns match definitions
3. Compare to original labels (should mostly overlap but with finer distinctions)

---

## Expected Outcomes

### Label Distribution

Starting from CaBuAr training set (~1,366 samples, ~17.8M pixels):

| Class | Expected % | Description | USGS Source |
|-------|-----------|-------------|---|
| **0 - Unburned** | 40-50% | No burn impact | MTBS Unburned |
| **1 - Low Severity** | 5-10% | Light scorch | MTBS Low |
| **2 - Moderate Severity** | 15-25% | Substantial change | MTBS Moderate |
| **3 - High Severity** | 10-20% | Severe damage | MTBS High |
| **4 - Extreme Severity** | 2-5% | Complete consumption | MTBS Extreme |
| **5 - Water** | 1-3% | Water bodies | ESA/MNDWI |
| **6 - Cloud/Shadow** | 2-5% | Invalid data | Sentinel-2 SCL |

### Benefits for Training

**For RGB+IR model**:
- ✓ Cleaner training signal (no "old scar" confusion)
- ✓ Multiple classes → model learns burn severity
- ✓ Better generalization to real-world burn imagery

**For transfer to NAIP**:
- ✓ If NAIP data has recent fires → model recognizes
- ✓ If NAIP data has old scars → model classifies appropriately
- ✓ Model can distinguish burn age → more useful for applications

---

## Implementation Plan

### Step 1: Spectral Analysis (Automated)
**Input**: CaBuAr dataset (train/val/test splits)  
**Process**: Compute NBR/dNBR, apply USGS thresholds, water/cloud detection  
**Output**: Multi-class label arrays [train+val+test tiles × 512 × 512]  
**Save**: `multi_class_labels_YYYYMMDD_HHMMSS.pt` (Drive)  
**Time**: ~30 min (full dataset analysis)  
**Tools**: `src/drive_utils.py` for Drive management

### Step 2: Quality Control (Manual Inspection)
**Input**: Multi-class labels, original binary labels  
**Process**: Sample ~20 tiles, verify class assignments, check spectral coherence  
**Verify**: Class distributions match expected, no collapse to single class  
**Output**: QC report, refined thresholds if needed  
**Save**: `qc_results_YYYYMMDD_HHMMSS.json` (Drive)  
**Time**: ~1 hour  
**Reference**: Tile 18 analysis as validation case

### Step 3: Model Preparation
**Input**: Cleaned multi-class labels (7 classes)  
**Process**: Extract RGB+IR (4 channels) from 12-band S2  
**Create**: Train/val/test loaders with multi-class labels  
**Output**: Training dataset in RGB+IR format  
**Save**: `rgb_ir_dataset_YYYYMMDD_HHMMSS.pt` (Drive)  
**Time**: ~15 min

### Step 4: Training (Phase 4)
**Model**: U-Net with 4-channel input (RGB+IR)  
**Task**: Multi-class burn severity (7 classes: 0-6)  
**Loss**: Cross-entropy with class weighting  
**Validation**: On cleaned multi-class labels  
**Metrics**: Per-class IoU, macro F1-score, confusion matrix  
**Save**: Checkpoints + training log (Drive)  
**Time**: ~2-3 hours GPU

### Step 5: Transfer to NAIP (Phase 5)
**Input**: Trained 4-channel model, NAIP imagery  
**Process**: Zero-shot inference (no fine-tuning)  
**Output**: Burn maps with age/severity classes  
**Validation**: Visual inspection, spatial coherence  
**Save**: Predictions + analysis (Drive)  
**Time**: ~30 min

---

## Google Drive Caching

**All intermediate products are timestamped and saved to Drive.**

See: `docs/phase3_relabeling/COLAB_DRIVE_SETUP.md`

**Automatic reconnection** if Drive disconnects during long runs.

**File structure**:
```
MyDrive/RETINNA_cache/
├── phase3/relabeling/
│   ├── multi_class_labels_YYYYMMDD_HHMMSS.pt
│   ├── metrics_YYYYMMDD_HHMMSS.json
│   ├── qc_results_YYYYMMDD_HHMMSS.json
│   └── config_YYYYMMDD_HHMMSS.json
└── phase4/rgb_ir_training/
    ├── checkpoint_YYYYMMDD_HHMMSS.pt
    ├── training_log_YYYYMMDD_HHMMSS.json
    └── metrics_YYYYMMDD_HHMMSS.json
```

---

## Success Criteria

### Phase 3 (Relabeling)
- ✓ Multi-class labels assigned to all 1,366 training samples
- ✓ Class distribution reasonable (>5% each major class)
- ✓ Spectral coherence verified for recent burn class
- ✓ Agreement with original labels >80% for "obvious" cases

### Phase 4 (RGB+IR Training)
- ✓ Model trains without error
- ✓ Per-class IoU > 0.40 for recent burn class
- ✓ No class collapse (model uses all 5 classes)
- ✓ Test set performance comparable to Phase 2 binary model

### Phase 5 (NAIP Transfer)
- ✓ Model generates predictions on NAIP data
- ✓ Predictions show meaningful variation (not all one class)
- ✓ Spatial coherence reasonable (not random noise)
- ✓ Can distinguish large vs small burn patches

---

## Open Questions

1. **Threshold refinement for Sentinel-2**: USGS MTBS thresholds were developed for Landsat (30m resolution). Should we adjust for Sentinel-2 (20m)?
   - Current approach: Apply as-is (conservative)
   - Alternative: Empirically validate on CaBuAr test set
   - Decision: Validate first run, adjust if needed based on QC results

2. **Other land cover classes**: Roads, structures, snow, ice?
   - **Roads/Structures**: Resolution limitation (Sentinel-2 is 20m, features often subpixel)
     - Status: Excluded (too unreliable, risk of false positives for bare soil/burns)
     - Recommendation: Use Class 6 (Other) if needed post-hoc
   - **Water**: Included (Class 5) ✓ Reliable with MNDWI
   - **Snow/Ice**: Detectable but rare in California fire season
     - Status: Include in SCL cloud class (Class 6) if detected by Sentinel-2 SCL
     - Recommendation: Monitor during QC, add if significant

3. **Handling edge cases**: Pixels at class boundaries (e.g., exactly at threshold)
   - Current approach: Use upper threshold (e.g., dnbr = 0.27 → Class 0, not Class 1)
   - Rationale: Conservative for fire detection (favor low severity over unburned)
   - Alternative: Add small epsilon buffer (dnbr > 0.27 + ε for strict assignment)

4. **NAIP validation without ground truth**: How to measure success?
   - Visual inspection of predicted maps (spatial coherence)
   - Comparison to USGS MTBS reference burns (if available for same areas)
   - Spatial statistics (size distribution of fire patches)
   - Domain expert review (fire personnel)
   - Recommendation: Use all four approaches

---

## References

### Official Standards

1. **USGS Monitoring Trends in Burn Severity (MTBS)**
   - https://www.mtbs.gov/
   - NBR thresholds for burn severity classification
   - Used nationally for fire monitoring and analysis

2. **USGS Remote Sensing Phenology Lab**
   - Burn severity classification methodology
   - dNBR threshold definitions

### Core Literature

3. **Key et al. (2006)** - "The Normalized Burn Ratio (NBR): A Landsat measure for measuring burn severity"
   - *Remote Sensing of Environment*, vol. 29
   - Original NBR formulation and justification

4. **Miller & Thode (2007)** - "Quantifying burn severity in a heterogeneous landscape with a relative version of the delta Normalized Burn Ratio (dNBR)"
   - *Remote Sensing of Environment*, vol. 109
   - Relative dNBR for Landsat/Sentinel-2 application

### Sentinel-2 Specific

5. **ESA Sentinel-2 Level-2A Product Specification**
   - Scene Classification Layer (SCL) band definitions
   - Cloud and shadow classification
   - https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi

6. **McFEE et al. (2013)** - "Water body mapping with Sentinel-2 data"
   - Modified Normalized Difference Water Index (MNDWI)
   - Water detection methodology

### Related

7. **Keeley (2009)** - "Fire intensity, fire severity and burn severity: A brief review and suggested usage"
   - *International Journal of Wildland Fire*, vol. 18
   - Conceptual framework for severity classification

---

**All class definitions and thresholds in this strategy are based on official USGS MTBS standards unless otherwise noted.**

---

## Next Steps

1. **Before implementation**: Review thresholds with domain knowledge
2. **Ready?** → Proceed to implementation in Phase 3 notebook
3. **Notebook**: `notebooks/03b_spectral_relabeling.ipynb` (to be created)

---

**Status**: ✅ Strategy approved, based on official USGS MTBS standards  
**Authority**: USGS Monitoring Trends in Burn Severity (MTBS) program  
**Owner**: Stephen Cerruti with Claude Code assistance  
**Date**: 2026-06-24  
**Last Updated**: 2026-06-24 (added water detection, official references, addressed class design)
