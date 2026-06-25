# Phase II: Spectral Relabeling & Model Training - Master Document

**Date**: 2026-06-24 to 2026-06-25  
**Status**: II_01 & II_03 ✅ Complete | II_02 Ready to Start  
**Owner**: RETINNA Investigation

---

## Executive Summary

**Problem**: CaBuAr dataset has binary administrative labels (Cal Fire fire perimeters), not spectral burn severity classifications. This misalignment prevents training a spectral model on spectral data.

**Solution**: Phase II creates multi-class burn severity labels using RdNBR (Relativized dNBR) calibrated to CaBuAr's arid California landscape, then trains U-Net on these improved labels.

**Result**: 848 training images with 7-class severity labels, metadata database of 298 fires, ready for U-Net training.

---

## Phase II_01: Spectral Relabeling ✅ COMPLETE

**Document**: [PHASE_II_01_COMPLETE.md](PHASE_II_01_COMPLETE.md)

### Objective
Replace binary CaBuAr labels with multi-class burn severity using spectral indices.

### Technical Approach

#### 1. Spectral Index Selection
- **Bands**: B08 (NIR, 20m) and B12 (SWIR-2, 20m) - industry standard for burn detection
- **Formula**: NBR = (NIR - SWIR) / (NIR + SWIR)
  - Healthy vegetation: NBR ≈ +0.5 to +0.9
  - Burned/dead: NBR ≈ -0.5 to -0.1

#### 2. Relativized dNBR (RdNBR) - Key Innovation
**Why RdNBR instead of absolute dNBR?**
- **Problem**: Absolute dNBR treats small changes in sparse vegetation as severe burns
- **Solution**: RdNBR = dNBR / sqrt(|NBRpre|) normalizes by pre-fire vegetation amount
- **Benefit**: Better distribution for arid CA landscape (Miller & Thode 2007)
- **Evidence**: CaBuAr's sparse chaparral matches RdNBR use case perfectly

#### 3. Per-Sample Calibration Algorithm
For each of 424 samples:
1. Compute RdNBR = NBRpre - NBRpost / sqrt(|NBRpre|)
2. Classify pixels using USGS MTBS RdNBR thresholds → ground truth
3. Analyze post-fire NBR distribution for each RdNBR class
4. Calibrate single-image NBR thresholds from RdNBR distribution
5. Apply calibrated thresholds to pre-fire NBR → pre-fire labels
6. Apply calibrated thresholds to post-fire NBR → post-fire labels

**Rationale**: 
- Uses RdNBR to establish what severity classes look like in THIS data
- Then applies those learned thresholds to single-image NBR
- Pre-fire shows baseline, post-fire shows fire impact
- Doubles training data: 424 samples → 848 images

### USGS MTBS Thresholds (Applied to RdNBR)
| Class | RdNBR Range | Description |
|-------|------------|-------------|
| Unburned | < 0.1 | No burn impact |
| Low Severity | 0.1-0.27 | Light burn, vegetation survives |
| Moderate Severity | 0.27-0.44 | Substantial vegetation loss |
| High Severity | 0.44-0.66 | Severe burn, canopy kill |
| Extreme Severity | ≥ 0.66 | Complete consumption |
| Water | MNDWI > 0.3 | Special land cover |
| Cloud/Shadow | Blue > 0.25 + Blue/NIR > 0.8 | Invalid pixels |

### Results: Class Distribution

**Training Set (278 samples → 278 pre + 278 post)**
```
Pre-fire:   95% Low Severity | 1% Extreme | 2% Water/Cloud
Post-fire:  94% Low Severity | 2% Extreme | 2% Water/Cloud
```

**Validation Set (78 samples → 156 images)**
```
Pre-fire:   84% Low Severity | 12% Extreme | 3% Water/Cloud
Post-fire:  81% Low Severity | 14% Extreme | 3% Water/Cloud
```

**Test Set (68 samples → 136 images)**
```
Pre-fire:   85% Low Severity | 0.03% Extreme | 8% Water/Cloud
Post-fire:  83% Low Severity | 0.91% Extreme | 8% Water/Cloud
```

### Key Findings
1. **Bimodal distribution**: Most pixels are either Low Severity (sparse, unburned) or Extreme (complete burn)
2. **Little middle ground**: Moderate/High classes are <1-2% (realistic for CA chaparral)
3. **Post > Pre confirmed**: Post-fire shows increased Extreme (QC ✓)
4. **Data reflects reality**: CA fires in sparse vegetation are extreme or minimal

### Implementation Details
- **File**: [notebooks/II_01_spectral_relabeling.ipynb](../notebooks/II_01_spectral_relabeling.ipynb)
- **Band indices (CaBuAr order)**:
  - B08 (NIR): index 7
  - B12 (SWIR): index 11
  - B03 (Green): index 2
  - B02 (Blue): index 1
- **Cloud detection**: Blue > 0.25 AND Blue/NIR > 0.8
- **Water detection**: MNDWI > 0.3
- **Vectorized classification**: ~3000× faster than pixel-by-pixel (4 sec vs 3.5 hrs)

### Output Artifacts
- **Labels**: `multi_class_labels_TIMESTAMP.pt` (424×512×512 → 848 images with pre/post)
- **Metrics**: `metrics_TIMESTAMP.json` (class distributions, split info)
- **Location**: Drive at `/phase2/II_01_relabeling/`

### Quality Assurance
- ✅ Band indices verified (B08/B12 correct)
- ✅ Spectral indices computed correctly (dNBR, RdNBR, MNDWI)
- ✅ Post-fire shows more severe than pre-fire
- ✅ Cloud/Water detection consistent across pre/post
- ✅ JSON serialization handles numpy int64 types

---

## Phase II_02: U-Net Training ⏳ READY

**Document**: (To be created)

### Objective
Train U-Net segmentation model on 848 labeled images with 7-class burn severity.

### Input Data
- **Images**: 848 Sentinel-2 images (424 samples × 2 for pre+post)
- **Labels**: 7-class severity predictions from II_01
- **Resolution**: 512×512 pixels @ 20m/pixel
- **Format**: PyTorch tensors (image + label pairs)

### Expected Implementation
```python
# Model: U-Net (from PA3)
# Input: 12-band Sentinel-2 (normalized)
# Output: 7-class severity prediction (512×512)
# Loss: Weighted cross-entropy (handle class imbalance)
# Metrics: Per-class IoU, pixel accuracy, confusion matrix
```

### Key Considerations
1. **Class imbalance**: 95% Low Severity vs 1-2% Extreme
   - Use weighted loss or focal loss
   - Monitor per-class metrics, not just overall accuracy
2. **Data augmentation**: Rotation, flipping, minor intensity shifts
3. **Validation strategy**: Use fold-based cross-validation from CaBuAr splits
4. **Hyperparameter tuning**: Learning rate, batch size, epochs

### Success Metrics
- Per-class IoU > 0.7 (target)
- Extreme Severity IoU > 0.5 (harder class)
- No overfitting (val loss tracks train loss)

---

## Phase II_03: Metadata Extraction ✅ COMPLETE

**Document**: [cabuaur_metadata.json](cabuaur_metadata.json)

### Objective
Extract and structure CaBuAr metadata for future enrichment with coordinates and dates.

### Implementation
- **File**: [notebooks/II_03_metadata_extraction.ipynb](../notebooks/II_03_metadata_extraction.ipynb)
- **Input**: CaBuAr HDF5 at `/RETINNA_DATA/cabuaur/512x512.hdf5`
- **Output**: `cabuaur_metadata.json` (246 KB)

### Extracted Data
```
cabuaur_metadata.json
├── metadata
│   ├── created: 2026-06-25T06:00:37
│   ├── total_fires: 298 unique UUIDs
│   ├── total_samples: 534 (includes test set)
│   └── note: ready for enrichment
└── fires[] (one per unique fire)
    ├── uuid: fire identifier
    ├── fire_name: null (to be filled from Cal Fire)
    ├── fire_id: null (to be filled from Cal Fire)
    ├── location: {bbox, center coords} (all null)
    ├── pre_date, post_date: null (to be filled from Sentinel-2)
    ├── split: {train: N, val: N}
    └── samples[]
        ├── sample_id (UUID_tileindex)
        ├── tile_idx (0-4 per fire)
        ├── fold (train/val/test)
        └── comments (metadata from HDF5)
```

### Data Quality Notes
- HDF5 contains: UUIDs, tile indices, folds, comments
- HDF5 NOT contain: dates, coordinates, fire names
- Comments array: unknown encoding (1-11 range, or -1)
- UUIDs: random, no spatial information embedded

### Future Enrichment Paths
1. **Dates**: Query Sentinel-2 API or metadata in original GeoTIFF
2. **Coordinates**: Map UUIDs to Cal Fire fire perimeters (bounding boxes)
3. **Fire names**: Cross-reference Cal Fire incident database by date/location
4. **Comments**: Investigate meaning (fire-specific flags? quality indicators?)

### Implementation Notes
- Handles numpy int64 → Python int conversion for JSON serialization
- Scales to 534 samples in <5 seconds
- Located at `/docs/cabuaur_metadata.json` and Drive

---

## Technical Decisions & Rationale

### 1. Why RdNBR over dNBR?
**Decision**: Use Relativized dNBR for arid landscapes  
**Rationale**: 
- CaBuAr contains California fires in sparse chaparral
- Absolute dNBR over-weights small changes in low-vegetation pixels
- Miller & Thode (2007) shows RdNBR performs better for heterogeneous landscapes
- Produces more realistic class distribution for this ecosystem
**Evidence**: Successfully generated bimodal distribution (sparse vs burned) rather than uniform skew

### 2. Why Per-Sample Calibration?
**Decision**: Calibrate NBR thresholds per fire using RdNBR as ground truth  
**Rationale**:
- USGS MTBS thresholds designed for "typical" landscapes
- CaBuAr data may not fit those assumptions
- Per-sample approach lets data speak: derive thresholds from actual distributions
- More generalizable than forcing fixed thresholds
**Evidence**: Prevents threshold mismatch that plagued earlier approaches

### 3. Why Pre and Post Both?
**Decision**: Generate labels for both pre-fire and post-fire images  
**Rationale**:
- Pre-fire: Baseline severity (shows what the landscape looked like before)
- Post-fire: Burn impact (shows fire damage)
- Doubles training data: 424 → 848 images
- Model learns from both perspectives
**Evidence**: Post should be more severe than pre (QC confirmed ✓)

### 4. Why 7 Classes Instead of Binary?
**Decision**: Multi-class (Unburned, Low, Moderate, High, Extreme, Water, Cloud)  
**Rationale**:
- USGS MTBS defines severity on a spectrum, not binary
- Fire recovery planning needs fine-grained severity
- More informative than binary (on/off)
- PA3 objective: implement semantic segmentation, not just binary classification
**Evidence**: Matches professional burn assessment standards

### 5. Why Spectral over Administrative Labels?
**Decision**: Replace Cal Fire perimeters with spectral-based labels  
**Rationale**:
- CaBuAr labels are administrative boundaries, not spectral reality
- Perimeters include unburned islands, miss low-severity patches
- Training a spectral model on administrative labels creates mismatch
- Aligns with ML best practice: validate ground truth before training
**Evidence**: Documented in [PHASE_II_PIVOT_JUSTIFICATION.md](PHASE_II_PIVOT_JUSTIFICATION.md)

---

## File Organization

```
RETINNA/
├── notebooks/
│   ├── II_01_spectral_relabeling.ipynb       [MAIN PROCESSING]
│   ├── II_02_unet_training.ipynb             [TO CREATE]
│   └── II_03_metadata_extraction.ipynb       [METADATA EXTRACTION]
├── docs/
│   ├── PHASE_II_MASTER.md                    [THIS FILE]
│   ├── PHASE_II_01_COMPLETE.md               [II_01 DETAILS]
│   ├── PHASE_II_PIVOT_JUSTIFICATION.md       [RATIONALE]
│   ├── OFFICIAL_S2_CLASSIFICATION_STANDARDS.md [USGS STANDARDS]
│   ├── cabuaur_metadata.json                 [METADATA DATABASE]
│   ├── DATA_LOADING_BUG_FIX.md               [SCL BAND ISSUE]
│   ├── PIXEL_CLASSIFICATION_OPTIMIZATION.md  [VECTORIZATION]
│   └── CABUAUR_ORIGINAL_PAPER_ANALYSIS.md    [DATASET PROVENANCE]
├── src/
│   ├── dataset.py                            [DATA LOADER]
│   └── unet.py                               [MODEL ARCHITECTURE]
└── analysis_artifacts/
    └── tile18_diagnostic.png                 [SPECTRAL ANALYSIS]
```

---

## Key Metrics & Statistics

### Dataset Composition
| Split | Samples | Images | Unburned | Low | Moderate | High | Extreme | Water | Cloud |
|-------|---------|--------|----------|-----|----------|------|---------|-------|-------|
| Train | 278 | 556 | 0.47% | 95% | 0.08% | 0.06% | 1.13% | 2.20% | 0.92% |
| Val | 78 | 156 | 0.22% | 84% | 0.04% | 0.01% | 12% | 3% | 1% |
| Test | 68 | 136 | 0.25% | 85% | 1.46% | 0.01% | 0.03% | 8% | 6% |

### Processing Performance
- **Spectral index computation**: ~0.1 sec per sample (vectorized)
- **Classification**: ~0.01 sec per sample (GPU-accelerated)
- **Total for 424 samples**: ~4 seconds
- **Improvement**: ~3000× faster than original pixel-by-pixel approach (3.5 hrs → 4 sec)

### Metadata
- **Unique fires**: 298 (in metadata database)
- **Total samples (w/ test)**: 534
- **Tiles per fire**: 1-4 (average 1.8)
- **Database size**: 246 KB JSON

---

## Dependencies & Tools

### Python Packages
```python
torch==2.0+           # PyTorch
torchvision           # Computer vision
torchgeo              # GeoSpatial datasets
h5py                  # HDF5 file handling
numpy                 # Numerical computing
matplotlib            # Visualization
google-colab          # Colab utilities (on Colab)
```

### External Data
- **CaBuAr dataset**: Hugging Face (512x512.hdf5, 26 GB)
- **USGS MTBS standards**: Public documentation
- **Sentinel-2 data**: Pre-packaged in CaBuAr

### Compute Requirements
- **GPU**: NVIDIA CUDA (optional but recommended)
  - Without GPU: ~30 sec per sample
  - With GPU: ~0.01 sec per sample
- **Storage**: 26 GB for CaBuAr cache
- **RAM**: 4+ GB recommended

---

## Next Steps

### Immediate (Phase II_02)
1. Create `II_02_unet_training.ipynb`
2. Implement U-Net with 7-class output
3. Train with weighted cross-entropy loss
4. Validate with fold-based cross-validation
5. Measure per-class IoU and confusion matrix

### Medium-term (Phase II_03 Enrichment)
1. Query Sentinel-2 metadata for pre/post dates
2. Map fire UUIDs to Cal Fire database
3. Extract bounding boxes from fire perimeters
4. Update metadata JSON with coordinates and fire names
5. Create spatial visualization of fires

### Long-term (Phase III)
1. Transfer learned model to NAIP data (Phase II_03_NAIP)
2. Generate burn predictions on new imagery
3. Compare spectral vs administrative labels
4. Document lessons learned for burn detection

---

## Validation Checklist

### II_01 ✅
- [x] Band indices verified (B08/B12)
- [x] RdNBR formula implemented correctly
- [x] USGS MTBS thresholds applied
- [x] Per-sample calibration working
- [x] Pre-fire baseline < post-fire severity (QC)
- [x] Cloud/water detection consistent
- [x] 848 training images generated
- [x] JSON serialization fixed (numpy types)

### II_02 (Pre-training)
- [ ] U-Net architecture matches PA3 requirements
- [ ] Data loader returns 12-band images + 7-class labels
- [ ] Weighted loss handles class imbalance
- [ ] Validation strategy uses fold assignments
- [ ] No data leakage (train/val/test splits respected)

### II_03 ✅
- [x] Metadata extracted from HDF5
- [x] 298 fires identified
- [x] 534 samples catalogued
- [x] JSON structure ready for enrichment
- [x] Database saved to docs/ and Drive

---

## References

### Academic Papers
1. **Key et al. (2006)** - Normalized Burn Ratio (NBR) for Landsat
2. **Miller & Thode (2007)** - Relativized dNBR for heterogeneous landscapes
3. **CaBuAr Paper (2024)** - California Burned Areas dataset
   - Cambrin, D.R., Colomba, L., Garza, P.
   - IEEE Geoscience and Remote Sensing Magazine, Vol. 11, No. 3
   - DOI: 10.1109/MGRS.2023.3292467

### Official Standards
- **USGS MTBS**: https://www.mtbs.gov/
- **ESA Sentinel-2 Handbook**: https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi

### Project Documentation
- [PHASE_II_01_COMPLETE.md](PHASE_II_01_COMPLETE.md) - Detailed II_01 results
- [OFFICIAL_S2_CLASSIFICATION_STANDARDS.md](OFFICIAL_S2_CLASSIFICATION_STANDARDS.md) - USGS standards + RdNBR
- [PHASE_II_PIVOT_JUSTIFICATION.md](PHASE_II_PIVOT_JUSTIFICATION.md) - Why Phase II over Phase I

---

## Contact & Questions

**Project Lead**: Stephen Cerruti  
**Code Review**: Claude Haiku 4.5  
**Status**: Active Development  
**Last Updated**: 2026-06-25

For questions on spectral processing, see II_01 documentation.  
For questions on metadata structure, see II_03 documentation.  
For questions on training strategy, check back after II_02 creation.

---

**Project Status**: Phase II 50% complete (II_01 ✅, II_02 ⏳, II_03 ✅)
