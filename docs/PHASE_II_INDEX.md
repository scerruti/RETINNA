# Phase II Documentation Index

**Phase II Status**: 50% Complete (II_01 ✅, II_02 ⏳, II_03 ✅)

---

## Core Documents

### [PHASE_II_MASTER.md](PHASE_II_MASTER.md) 📋 START HERE
**Comprehensive overview of the entire Phase II pipeline**
- Executive summary of problem and solution
- II_01 spectral relabeling (complete)
- II_02 U-Net training (roadmap)
- II_03 metadata extraction (complete)
- Technical decisions and rationale
- Validation checklist
- File organization
- Next steps

---

## Phase II_01: Spectral Relabeling

### [PHASE_II_01_COMPLETE.md](PHASE_II_01_COMPLETE.md)
**Detailed results and findings from spectral relabeling**
- Objective and approach (RdNBR calibration)
- Results by dataset split (Train/Val/Test)
- Class distributions (showing bimodal pattern)
- Quality control metrics
- Files generated
- Lessons learned

### [OFFICIAL_S2_CLASSIFICATION_STANDARDS.md](OFFICIAL_S2_CLASSIFICATION_STANDARDS.md)
**USGS MTBS standards and implementation guide**
- Absolute dNBR thresholds (0.1, 0.27, 0.44, 0.66)
- Relativized dNBR (RdNBR) for arid regions
- Band recommendations (B08/B12 standard)
- Algorithm pseudocode
- Expected class distributions
- QC validation procedures
- References (Key et al., Miller & Thode)

### [PIXEL_CLASSIFICATION_OPTIMIZATION.md](PIXEL_CLASSIFICATION_OPTIMIZATION.md)
**Performance optimization from pixel-loop to vectorization**
- Original pixel-by-pixel algorithm (educational)
- Vectorized tensor operations (production)
- Performance comparison (3.5 hrs → 4 sec)
- Algorithm equivalence proof
- Decision rationale for vectorization
- Preserved original for reference

### [DATA_LOADING_BUG_FIX.md](DATA_LOADING_BUG_FIX.md)
**Documentation of SCL band issue and workaround**
- SCL band normalization problem (floats instead of class codes)
- Why spectral cloud detection was needed
- Blue/NIR ratio method for cloud detection
- Band index corrections (B08/B12 vs B07/B8A)

---

## Phase II_02: U-Net Training

### [PHASE_II_02_CHANGE_DETECTION_STRATEGY.md](PHASE_II_02_CHANGE_DETECTION_STRATEGY.md)
**Architecture decision: Change-detection input for Phase III transfer**
- Why difference-based input (Post - Pre)
- NAIP-compatible band selection (4 channels)
- Coherence with Phase II_01 RdNBR labels
- Trade-offs and Phase III transfer strategy

### Implementation: [II_02_unet_training.ipynb](../notebooks/II_02_unet_training.ipynb)
**U-Net training notebook (ready to run on Colab)**
- Change-detection dataset loader
- U-Net architecture (4 → 7 classes)
- Weighted cross-entropy loss for class imbalance
- Training loop with validation
- Model saving and Phase III readiness verification

---

## Phase II_03: Metadata Extraction

### [cabuaur_metadata.json](cabuaur_metadata.json)
**Structured metadata database of 298 fires and 534 samples**
- 298 unique fire UUIDs
- 534 total samples (includes test set)
- Train/Val/Test split information
- Tile indices per fire
- Comments metadata from HDF5
- Structure ready for enrichment:
  - fire_name (null → to be filled)
  - fire_id (null → to be filled)
  - location bbox and center coords (null → to be filled)
  - pre_date, post_date (null → to be filled)

---

## Supporting Documentation

### [PHASE_II_PIVOT_JUSTIFICATION.md](PHASE_II_PIVOT_JUSTIFICATION.md)
**Defense of Phase II approach vs returning to Phase I**
- PA3 learning objectives coverage
- Alignment with professional ML practices
- Comparison to Magnifier paper approach
- Real-world industry precedents
- What to say to teacher

### [CABUAUR_ORIGINAL_PAPER_ANALYSIS.md](CABUAUR_ORIGINAL_PAPER_ANALYSIS.md)
**Analysis of CaBuAr dataset provenance and limitations**
- Ground truth source (Cal Fire perimeters, not spectral)
- Dataset structure and splits
- Why spectral relabeling was necessary
- Magnifier paper follow-up work (2025)
- Research implications

---

## Notebooks (Executable Code)

### [II_01_spectral_relabeling.ipynb](../notebooks/II_01_spectral_relabeling.ipynb)
**Main processing notebook for spectral label generation**
- Loads CaBuAr dataset from Google Drive
- Computes RdNBR and calibrates thresholds
- Generates 7-class labels for all samples
- Validates QC metrics
- Saves to Drive: `multi_class_labels_TIMESTAMP.pt` + metrics

### [II_03_metadata_extraction.ipynb](../notebooks/II_03_metadata_extraction.ipynb)
**Metadata extraction and structuring**
- Extracts HDF5 metadata (UUIDs, folds, comments)
- Converts numpy types to JSON-serializable Python types
- Builds structured database
- Saves `cabuaur_metadata.json` to Drive and docs/

### [II_02_unet_training.ipynb](../notebooks/II_02_unet_training.ipynb)
**To be created** - will implement U-Net training pipeline

---

## Output Artifacts

### Generated in Phase II_01
- **multi_class_labels_TIMESTAMP.pt** (424×512×512 → 848 images)
  - Location: Google Drive `/phase2/II_01_relabeling/`
  - Content: 7-class severity predictions for all samples
  
- **metrics_TIMESTAMP.json** (class distributions)
  - Location: Google Drive `/phase2/II_01_relabeling/`
  - Content: Per-split class statistics

### Generated in Phase II_03
- **cabuaur_metadata.json** (298 fires, 534 samples)
  - Location: `/docs/cabuaur_metadata.json` (local) and Google Drive
  - Content: Structured metadata ready for enrichment

---

## Quick Stats

### Phase II_01 Results
- **Input**: 424 samples × 2 (pre+post) = 848 images
- **Output**: 7-class severity labels
- **Classes**: Unburned, Low, Moderate, High, Extreme, Water, Cloud
- **Processing time**: 4 seconds (GPU-accelerated)
- **Performance**: 3000× faster than original (3.5 hrs → 4 sec)

### Phase II_03 Results
- **Unique fires**: 298
- **Total samples**: 534
- **Database size**: 246 KB JSON
- **Metadata completeness**: 
  - ✅ UUIDs, tile indices, folds, comments
  - ❌ Fire names, dates, coordinates (future work)

---

## Reading Guide

**New to Phase II?**
1. Start: [PHASE_II_MASTER.md](PHASE_II_MASTER.md) (15 min overview)
2. Deep dive: [PHASE_II_01_COMPLETE.md](PHASE_II_01_COMPLETE.md) (technical details)
3. Justification: [PHASE_II_PIVOT_JUSTIFICATION.md](PHASE_II_PIVOT_JUSTIFICATION.md) (why we chose this approach)

**Want to understand spectral standards?**
1. [OFFICIAL_S2_CLASSIFICATION_STANDARDS.md](OFFICIAL_S2_CLASSIFICATION_STANDARDS.md) (USGS thresholds + RdNBR)
2. [CABUAUR_ORIGINAL_PAPER_ANALYSIS.md](CABUAUR_ORIGINAL_PAPER_ANALYSIS.md) (dataset context)

**Ready to train U-Net?**
1. See [PHASE_II_MASTER.md#phase-ii_02-u-net-training--ready](PHASE_II_MASTER.md#phase-ii_02-u-net-training--ready) for requirements
2. Check metadata: [cabuaur_metadata.json](cabuaur_metadata.json) for data organization

**Troubleshooting?**
- Band indices wrong? → [DATA_LOADING_BUG_FIX.md](DATA_LOADING_BUG_FIX.md)
- Why vectorized classification? → [PIXEL_CLASSIFICATION_OPTIMIZATION.md](PIXEL_CLASSIFICATION_OPTIMIZATION.md)
- Class distribution odd? → [PHASE_II_01_COMPLETE.md#key-observations](PHASE_II_01_COMPLETE.md#key-observations)

---

## File Organization

```
RETINNA/docs/
├── PHASE_II_MASTER.md                           ← START HERE
├── PHASE_II_INDEX.md                            ← THIS FILE
├── PHASE_II_01_COMPLETE.md                      [Detailed II_01 results]
├── PHASE_II_PIVOT_JUSTIFICATION.md              [Why this approach]
├── OFFICIAL_S2_CLASSIFICATION_STANDARDS.md      [USGS standards + RdNBR]
├── PIXEL_CLASSIFICATION_OPTIMIZATION.md         [Vectorization explained]
├── DATA_LOADING_BUG_FIX.md                      [SCL band issue]
├── CABUAUR_ORIGINAL_PAPER_ANALYSIS.md           [Dataset provenance]
└── cabuaur_metadata.json                        [Metadata database]

RETINNA/notebooks/
├── II_01_spectral_relabeling.ipynb              [COMPLETE]
├── II_03_metadata_extraction.ipynb              [COMPLETE]
└── II_02_unet_training.ipynb                    [TO CREATE]
```

---

## Status Summary

| Phase | Status | Key Docs | Output |
|-------|--------|----------|--------|
| **II_01** | ✅ Complete | [01_COMPLETE.md](PHASE_II_01_COMPLETE.md) | 848 labeled images |
| **II_02** | ⏳ Ready | [MASTER.md#02](PHASE_II_MASTER.md#phase-ii_02-u-net-training--ready) | (TBD) |
| **II_03** | ✅ Complete | [metadata.json](cabuaur_metadata.json) | 298 fires catalogued |

---

**Last Updated**: 2026-06-25  
**Created By**: Claude Haiku 4.5  
**For Questions**: Refer to appropriate section or cross-reference documents
