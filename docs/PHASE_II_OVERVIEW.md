# Phase II: Cross-Sensor Transfer Learning

**Status**: ✅ Planning complete, implementation underway  
**Start Date**: 2026-06-24  
**Motivation**: Ground truth labels contain old burn scars; need multi-class spectral relabeling

---

## Why Phase II?

**Phase I Discovery** (Day 3-4):
- Built baseline U-Net (IoU 0.5609) ✓
- Tested class weighting (pos_weight=1.5) ✓
- **Found**: Ground truth is ambiguous (old scars labeled as recent burns)
- **Conclusion**: Can't optimize binary classifier on broken labels

**Phase II Solution**:
- Use USGS official burn severity standards
- Create multi-class labels via spectral analysis
- Train RGB+IR model (4 channels, NAIP-compatible)
- Transfer zero-shot to real NAIP imagery

---

## Notebooks

### II_01: Spectral Relabeling
**File**: `notebooks/II_01_spectral_relabeling.ipynb`  
**Input**: CaBuAr binary labels (burned/unburned)  
**Output**: Multi-class labels (7 classes) based on USGS MTBS  
**Time**: ~1 hour  
**Key Feature**: Automatic Drive reconnection, timestamped saves

**Classes** (USGS standard):
```
0: Unburned (dNBR ≥ 0.27)
1: Low Severity (0.05 < dNBR ≤ 0.27)
2: Moderate Severity (-0.1 < dNBR ≤ 0.05)
3: High Severity (-0.27 < dNBR ≤ -0.1)
4: Extreme Severity (dNBR < -0.27)
5: Water (MNDWI > 0.3)
6: Cloud/Shadow (spectral detection: Blue > 0.25 AND Blue/NIR > 0.8)
```

**⚠️ Note**: CaBuAr dataset does not include Sentinel-2 SCL band in usable form (see [DATA_LOADING_BUG_FIX.md](phase2_investigation/DATA_LOADING_BUG_FIX.md)). Cloud detection uses spectral thresholds instead, which generalizes to NAIP and other sensors.

**Metrics Saved**:
- Class distribution per split
- NBR/dNBR/MNDWI statistics
- Verification report

---

### II_02: RGB+IR Training (Planned)
**File**: `notebooks/II_02_rgb_ir_training.ipynb` (to be created)  
**Input**: Multi-class labels from II_01  
**Process**: Extract RGB+IR from 12-band Sentinel-2  
**Model**: U-Net with 4-channel input  
**Output**: Trained model checkpoint  
**Time**: ~2-3 hours GPU

---

### II_03: NAIP Transfer (Planned)
**File**: `notebooks/II_03_naip_transfer.ipynb` (to be created)  
**Input**: Trained RGB+IR model, NAIP imagery  
**Process**: Zero-shot inference (no fine-tuning)  
**Output**: Burn maps with severity classes  
**Time**: ~30 min

---

## Documentation

### Strategy
- `docs/phase3_relabeling/SPECTRAL_RELABELING_STRATEGY.md`
  - USGS MTBS standards and thresholds
  - Class definitions with spectral indicators
  - Implementation algorithm
  - Success criteria

### Drive Management
- `docs/phase3_relabeling/COLAB_DRIVE_SETUP.md`
  - `src/drive_utils.py` usage guide
  - Reconnection logic
  - Timestamped file management
  - Troubleshooting

### Pivot Justification
- `docs/phase2_investigation/PROJECT_PIVOT_JUSTIFICATION.md`
  - Why binary labels fail (Tile 18 evidence)
  - Visual proof (spectral difference analysis)
  - Why hyperparameter tuning can't fix it

---

## Key Differences from Phase I

| Aspect | Phase I | Phase II |
|--------|---------|----------|
| **Task** | Binary classification | Multi-class severity |
| **Input channels** | 24 (bi-temporal) | 4 (RGB+IR only) |
| **Labels** | Binary (broken) | 7-class USGS standard |
| **Validation** | On CaBuAr | On cleaned labels |
| **Test** | CaBuAr test set | NAIP zero-shot |
| **Goal** | Optimize on existing labels | Build correct labels first |

---

## Data Flow

```
Phase I (Complete)
├── I_01: Data pipeline
├── I_02: Exploratory analysis
├── I_03: Training (pos_weight=1.5)
├── I_04: Inference on test set
├── I_05: Analysis
└── Discovery: Labels are ambiguous

Phase II (In Progress)
├── II_01: Spectral relabeling ← YOU ARE HERE
│   ├── Compute NBR/dNBR/MNDWI
│   ├── Apply USGS thresholds
│   └── Generate 7-class labels
├── II_02: RGB+IR training
│   ├── Extract 4-channel input
│   ├── Train U-Net multi-class
│   └── Evaluate on cleaned labels
└── II_03: NAIP transfer
    ├── Load trained model
    ├── Inference on NAIP
    └── Validate generalization
```

---

## Success Criteria

### II_01 (Relabeling)
- ✓ Multi-class labels generated for all samples
- ✓ Class distribution reasonable (>1% each major class)
- ✓ No collapse to single class
- ✓ Verified files saved to Drive

### II_02 (RGB+IR Training)
- ✓ Model trains without error
- ✓ Per-class IoU > 0.35 for burn classes
- ✓ No class collapse
- ✓ Validation metrics improve from baseline

### II_03 (NAIP Transfer)
- ✓ Predictions generated on NAIP data
- ✓ Spatial coherence (not random noise)
- ✓ Meaningful class variation
- ✓ Comparison to reference burns if available

---

## Authority & Standards

All class definitions based on:
- **USGS Monitoring Trends in Burn Severity (MTBS)**
- **US Forest Service burn severity standards**
- **Key et al. (2006)** - Original NBR formulation
- **ESA Sentinel-2 documentation** - For water/cloud detection

---

## Learning Outcomes Alignment

✓ **Data quality assessment** — Recognizing and fixing mislabeled data  
✓ **Spectral analysis** — Using multiple bands for quantitative decisions  
✓ **Multi-class classification** — Beyond binary to meaningful categories  
✓ **Transfer learning** — Testing generalization across sensors  
✓ **Scientific rigor** — Investigation drives design (not vice versa)  

---

## Timeline

| Task | Est. Time | Status |
|------|-----------|--------|
| II_01: Spectral relabeling | 1 hour | Ready to run |
| II_02: RGB+IR training | 2-3 hours | Next |
| II_03: NAIP transfer | 30 min | Then |
| QC & Analysis | 1-2 hours | Final |
| **Total** | **~5-8 hours** | **Planned** |

---

## Next Steps

1. **Run II_01** on Colab (this notebook)
2. **Review metrics** — Check class distributions make sense
3. **Verify Drive saves** — Ensure timestamped files are accessible
4. **Proceed to II_02** — Train on cleaned multi-class labels

---

**Phase II represents a complete rethinking based on data discovery. We're not optimizing a broken model — we're building the correct data foundation first.**

---

**Created**: 2026-06-24  
**By**: Stephen Cerruti with Claude Code  
**Based on**: USGS MTBS standards + Phase I investigation findings
