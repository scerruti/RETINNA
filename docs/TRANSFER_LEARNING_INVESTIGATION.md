# Transfer Learning Investigation: Sentinel-2 → NAIP

Investigating cross-satellite transfer learning: train burn detection on Sentinel-2 RGB-IR, transfer to higher-resolution NAIP imagery.

**Issue**: #23  
**Sprint**: Option B (Feature Engineering & Transfer Learning)  
**Status**: Planning & Data Setup  

## Motivation

### The Challenge
- **Sentinel-2**: 10m resolution, 24 spectral bands, global coverage
- **NAIP**: 1m resolution, 4 bands (RGB + NIR), US-only, higher detail
- **Goal**: Use Sentinel-2's multi-spectral richness to create accurate labels, then transfer simpler 4-channel model to high-resolution NAIP

### Why This Matters
1. **Label Quality**: 24-channel Sentinel-2 provides richer spectral information for accurately identifying/verifying burn scars
2. **Reduced NAIP Labeling**: Transfer learned features from S2 to NAIP, requiring fewer manually labeled NAIP scenes
3. **Resolution Benefit**: NAIP's 1m resolution captures finer burn scar boundaries than S2's 10m
4. **Band Compatibility**: 4-channel RGB-IR structure matches NAIP's 4-band format for efficient transfer
5. **Practical Application**: Minimize expensive NAIP labeling while gaining high-resolution burn detection

### Hypothesis
> Use 24-channel Sentinel-2's spectral richness to accurately define burn masks. Train 4-channel RGB-IR model on those verified labels. Transfer to NAIP (1m resolution) with minimal additional labeling, leveraging Sentinel-2-learned features while gaining boundary precision.

## Data Setup

### Current State
- ✅ **CaBuAr Dataset**: Sentinel-2 24-channel (11 bands + cloud) × 2 timesteps = ~4400 samples
- ✅ **Cached on Drive**: `/content/drive/MyDrive/RETINNA_DATA/cabuaur/512x512.hdf5`
- ✅ **Native Splits**: train/val/test already provided (3098/644/644)
- ✅ **Baseline Model**: Trained on full 24 channels (IoU: 0.47)

### Data Extraction: RGB-IR Subset

**Goal**: Extract 4-channel RGB-IR from CaBuAr for simpler transfer learning baseline

**Bands Selected**:
| Index | Band Name | Resolution | Purpose |
|-------|-----------|-----------|---------|
| B02 | Blue | 10m | Visible spectrum |
| B03 | Green | 10m | Visible spectrum |
| B04 | Red | 10m | Visible spectrum |
| B08 | NIR | 10m | Vegetation/burn detection (key band) |

**Why these bands:**
- **RGB (B02, B03, B04)**: Direct visual match to NAIP's visible bands
- **NIR (B08)**: Critical for burn detection (burned areas show low NIR reflectance)
- **4 channels total**: Matches NAIP's 4-band structure (R, G, B, NIR)
- **Simpler model**: 4 channels vs 24 → faster training, easier transfer

### Data Workflow

```
Step 1: Extract 4-Channel RGB-IR from S2 (Band-Compatible with NAIP)
CaBuAr HDF5 (Sentinel-2 24-channel + TorchGeo masks)
├── 3098 train samples [2, 12, 512, 512]
├── 644 val samples [2, 12, 512, 512]
└── 644 test samples [2, 12, 512, 512]
(with existing TorchGeo ground truth masks)

Extract RGB-IR [B02, B03, B04, B08]
         ↓
CaBuAr RGB-IR Dataset
├── 3098 train samples [2, 4, 512, 512]
├── 644 val samples [2, 4, 512, 512]
└── 644 test samples [2, 4, 512, 512]
(same TorchGeo labels, 4 bands)

Step 2: Train 4-Channel RGB-IR Model on Sentinel-2
Train U-Net on 4-channel RGB-IR
         ↓
RGB-IR Model Checkpoint
(pre-trained on Sentinel-2, band-compatible with NAIP)

Step 3: Transfer to NAIP (Future)
NAIP imagery (1m resolution, 4 bands: R, G, B, NIR)
         ↓
Label subset of NAIP scenes (minimal effort)
         ↓
Fine-tune S2 RGB-IR model on NAIP
         ↓
Compare vs NAIP-only baseline
```

**Key point**: Uses existing TorchGeo CaBuAr masks (no label verification step). The 4-channel RGB-IR structure provides band-compatible representation for transfer to NAIP.

### Storage Plan

**Extracted Data Location**:
```
/content/drive/MyDrive/RETINNA_DATA/
├── cabuaur/                    (existing)
│   └── 512x512.hdf5           (24-channel, ~6GB)
│
└── cabuaur_rgbir/             (new)
    ├── cabuaur_rgbir.hdf5     (4-channel RGB-IR, ~1GB)
    ├── train_indices.npy      (3098 samples)
    ├── val_indices.npy        (644 samples)
    └── test_indices.npy       (644 samples)
```

**Why HDF5:**
- Same format as original (consistent)
- Efficient compression (~1GB for 4 channels vs 6GB for 24)
- Random access via TorchGeo
- Hierarchical structure for metadata

## Transfer Learning Approach

### Phase 1: RGB-IR Extraction (Current - Issue #23)
1. **Create extraction pipeline** (Colab notebook)
   - Load CaBuAr HDF5
   - Extract [B02, B03, B04, B08] for each sample
   - Preserve splits (train/val/test)
   - Save as new HDF5 on Drive

2. **Validate extraction**
   - Confirm shapes: [2, 4, 512, 512] (timesteps, channels, spatial)
   - Verify normalization (0-1 range after ÷10000)
   - Check band statistics (sanity check)

3. **Create RGB-IR DataLoader**
   - Reuse CaBuArDataset pattern
   - Support band selection (optional)
   - Integrate with existing training pipeline

### Phase 2: RGB-IR Training (Future)
1. **Train U-Net on RGB-IR**
   - Modify model: in_channels=4 (instead of 24)
   - Same training approach as baseline
   - Target: IoU ~0.40-0.45 (expect slight decrease from 24→4 channels)
   - Save checkpoint to Drive

2. **Analyze performance**
   - Compare RGB-IR IoU vs baseline (24-channel)
   - Document feature importance
   - Investigate which bands matter most

### Phase 3: NAIP Transfer Learning (Future)
1. **Acquire NAIP data**
   - Find burn scar NAIP scenes
   - Ensure available with labels
   - Create NAIP dataset wrapper

2. **Transfer learning setup**
   - Load RGB-IR checkpoint
   - Fine-tune on NAIP (small learning rate)
   - Freeze early layers, train later layers
   - Compare to NAIP-only training

3. **Evaluate transfer learning**
   - IoU on NAIP with pre-training vs without
   - Quantify improvement
   - Investigate what transferred vs what didn't

## Implementation Plan

### Phase 1: Extract RGB-IR from Sentinel-2 (This Week)
**Task**: `extract_rgbir.ipynb` (Colab notebook)

**What it does**:
- Load CaBuAr Sentinel-2 with existing TorchGeo masks
- Extract 4-channel RGB-IR [B02, B03, B04, B08] (band-compatible with NAIP)
- Create new HDF5 dataset with 4-channel data
- Preserve original train/val/test splits and masks

**Pseudocode**:
```python
# Load CaBuAr with existing TorchGeo masks
cabuaur = CaBuAr(root=..., split='all', download=True)

# Extract RGB-IR for each sample
for sample in cabuaur:
    image = sample['image']  # [2, 12, 512, 512] - 24 channels
    mask = sample['mask']     # [1, 512, 512] - TorchGeo ground truth
    
    # Select bands: B02(1), B03(2), B04(3), B08(7)
    rgbir = image[:, [1, 2, 3, 7], :, :]  # [2, 4, 512, 512]
    
    # Store with same mask in new HDF5
    save_to_hdf5(rgbir, mask, sample_id, split)
```

**Output**: `cabuaur_rgbir.hdf5` on Drive (TorchGeo labels, 4 channels)

### Phase 2: Train 4-Channel RGB-IR Model
**Task**: Modify `03_training.ipynb` to use RGB-IR dataset

**Changes**:
- Load RGB-IR dataset instead of 24-channel
- Update model: `UNet(in_channels=4, out_channels=2)`
- Use same training approach (20 epochs)
- Compare metrics to 24-channel baseline
- Save checkpoint for transfer learning

**Expected result**: Model trained on 4-channel S2, ready for NAIP transfer

### Phase 3: NAIP Transfer Learning (Future)
**Task**: Create `naip_transfer_learning.ipynb`

**Workflow**:
1. Acquire labeled NAIP burn scar data
2. Load RGB-IR checkpoint (from Phase 2)
3. Fine-tune on NAIP with low learning rate
4. Reduce earlier layers' learning rate (frozen) vs later layers
5. Evaluate: Does S2 pre-training help vs NAIP-only training?

## Success Criteria

### Phase 1 (Extraction)
- ✅ RGB-IR HDF5 successfully created on Drive
- ✅ Shapes correct: [2, 4, 512, 512] per sample
- ✅ All 3098+644+644 samples extracted
- ✅ Normalization verified (0-1 range)
- ✅ DataLoader works with new format

### Phase 2 (RGB-IR Training)
- ✅ Model trains without errors on 4 channels
- ✅ Validation IoU >= 0.35 (expect some degradation from 24→4)
- ✅ Checkpoint saved to Drive
- ✅ Comparable convergence to baseline

### Phase 3 (NAIP Transfer)
- ✅ NAIP dataset loaded successfully
- ✅ Transfer learning (fine-tuning) runs
- ✅ Pre-training improves NAIP IoU vs random init
- ✅ Feature transfer documented

## Technical Considerations

### Band Selection Rationale

**Why RGB-IR and not other combinations?**

| Combination | Channels | Reasoning |
|-------------|----------|-----------|
| RGB only | 3 | Too limited; misses vegetation change (no NIR) |
| **RGB-IR** | **4** | **Matches NAIP structure; includes burn-detection key (NIR)** |
| SWIR + NIR | 2 | Missing visible spectrum; harder for visual interpretation |
| All 12 bands | 12 | Too complex for transfer; harder to match to NAIP |
| All 24 channels | 24 | Current baseline; too many bands for NAIP transfer |

**NIR as critical band**:
- Burned vegetation: very low NIR reflectance
- Healthy vegetation: high NIR reflectance
- NDVI = (NIR - Red) / (NIR + Red) is standard burn index
- Without NIR, burn detection becomes much harder

### Normalization
- Sentinel-2 L2A standard: divide by 10000 (maps DN to reflectance 0-1)
- Both Sentinel-2 and NAIP use similar reflectance scale
- Ensures RGB-IR model can transfer without retraining normalization

### Model Architecture
- Same U-Net, just different input channels
- No architectural changes needed (fully convolutional)
- Skip connections still effective at 4 channels

## Expected Outcomes

### Best Case
- RGB-IR extraction works smoothly
- RGB-IR model achieves ~0.40 IoU (acceptable given 24→4 channel reduction)
- NAIP transfer learning shows significant improvement vs NAIP-only baseline
- Demonstrates value of S2 pre-training in reducing NAIP labeling effort
- Validates cross-satellite feature transfer with band-compatible structure

### Realistic Case
- RGB-IR extraction works, minor debugging needed
- RGB-IR model achieves ~0.35-0.38 IoU (channel impact moderate)
- NAIP transfer shows measurable but modest improvement
- Confirms S2 features transfer, but NAIP's high resolution requires some task-specific tuning
- Reduces NAIP labeling burden by ~30-50%

### Challenging Case
- RGB-IR model underperforms (IoU < 0.30)
- NAIP transfer provides minimal benefit
- Suggests 4-channel limitation or spectral mismatch
- Informs decision: may need full 24-channel transfer or more NAIP labels
- Guides Option C scope: cross-satellite transfer complexity

## Timeline

**Phase 1**: 2-3 hours (extraction + validation)  
**Phase 2**: 1-2 hours (training, mostly waiting for compute)  
**Phase 3**: 3-4 hours (NAIP setup + transfer learning)  

**Total**: 6-9 hours (spread across multiple sessions)

## Files & Artifacts

**To Create**:
- `extract_rgbir.ipynb` — Extraction notebook
- `src/dataset_rgbir.py` — RGB-IR DataLoader (optional)
- `RETINNA_DATA/cabuaur_rgbir/` — Extracted dataset
- Checkpoints for RGB-IR trained model
- NAIP transfer learning notebook (Phase 3)

**To Document**:
- RGB-IR vs baseline comparison
- Transfer learning results
- Lessons learned for cross-satellite generalization

## References

- Original paper: U-Net architecture and transfer learning patterns
- Related work: Cross-satellite domain adaptation in remote sensing
- Burn detection: Role of NIR and SWIR bands in detection accuracy

---

**Issue**: #23 (This document)  
**Status**: ✅ Planning Complete → Ready for Phase 1 Implementation  
**Next Step**: Build `extract_rgbir.ipynb` notebook for RGB-IR extraction
