# Validation Strategy for Option B: Multi-Class Pre-Training

**Purpose**: Empirically validate that multi-class pre-training improves burn detection over direct binary training

**Status**: Pre-implementation planning  
**To be completed**: After Option B experiments run

---

## Overview

Option B adds complexity (multi-class pre-training + fine-tuning). We must validate it actually works.

**Key question**: Does Model B (multi-class pre-train → fine-tune for burns) outperform Model A (direct binary training)?

**Answer method**: A/B testing with empirical metrics.

---

## Part 1: Validate Spectral Indices (Correctness)

### 1.1 Compute Spectral Indices

**What to compute** (from Sentinel-2 12 bands):
- **NDVI** = (Band 8 - Band 4) / (Band 8 + Band 4)  [Vegetation]
- **NDBI** = (Band 11 - Band 8) / (Band 11 + Band 8)  [Built-up/Urban]
- **NDWI** = (Band 3 - Band 11) / (Band 3 + Band 11)  [Water]
- **NBR** = (Band 8 - Band 12) / (Band 8 + Band 12)  [Burn Ratio]

**Compute on**: Pre-fire images from 100 random CaBuAr samples

**Code location**: `src/spectral_indices.py`

### 1.2 Validate Against Published Ranges

**Expected ranges from remote sensing literature:**

| Index | Min | Typical Vegetation | Max | Source |
|-------|-----|-------------------|-----|--------|
| NDVI | -1.0 | 0.4–0.8 | +1.0 | Rouse et al. 1973 |
| NDBI | -1.0 | 0.0–0.3 (urban) | +1.0 | Zha et al. 2003 |
| NDWI | -1.0 | 0.3–0.6 (water) | +1.0 | McFeeters 1996 |
| NBR | -1.0 | 0.4–0.9 (vegetation) | +1.0 | Key & Benson 2006 |

**Validation checks**:
```python
def validate_spectral_indices(sample_size=100):
    """Validate indices are in expected ranges"""
    
    for i in range(sample_size):
        sample = dataset[i]
        pre_fire = sample['image'][0]  # First temporal pass (pre-fire)
        
        ndvi = compute_ndvi(pre_fire)
        ndbi = compute_ndbi(pre_fire)
        ndwi = compute_ndwi(pre_fire)
        nbr = compute_nbr(pre_fire)
        
        # Range checks
        assert -1.0 <= ndvi.min() and ndvi.max() <= 1.0, f"NDVI out of range: {ndvi.min():.3f} to {ndvi.max():.3f}"
        assert -1.0 <= ndbi.min() and ndbi.max() <= 1.0, f"NDBI out of range"
        assert -1.0 <= ndwi.min() and ndwi.max() <= 1.0, f"NDWI out of range"
        assert -1.0 <= nbr.min() and nbr.max() <= 1.0, f"NBR out of range"
        
        # Sanity checks on means
        if ndvi.mean() < -0.2 or ndvi.mean() > 0.9:
            print(f"⚠️  Sample {i}: NDVI mean {ndvi.mean():.3f} outside typical range")
        if ndbi.mean() > 0.5:
            print(f"⚠️  Sample {i}: NDBI mean {ndbi.mean():.3f} suggests all urban (unlikely)")
    
    print("✅ All indices pass range validation")
```

### 1.3 Visualize Index Distributions

**Generate plots** (save to `results/validation/`):

```python
def plot_index_distributions(sample_size=500):
    """Plot histograms of spectral indices across dataset"""
    
    ndvi_all = []
    ndbi_all = []
    ndwi_all = []
    nbr_all = []
    
    for i in range(sample_size):
        sample = dataset[i]
        pre_fire = sample['image'][0]
        ndvi_all.append(compute_ndvi(pre_fire).flatten())
        ndbi_all.append(compute_ndbi(pre_fire).flatten())
        ndwi_all.append(compute_ndwi(pre_fire).flatten())
        nbr_all.append(compute_nbr(pre_fire).flatten())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].hist(np.concatenate(ndvi_all), bins=50, alpha=0.7, edgecolor='k')
    axes[0,0].set_title("NDVI Distribution (Expected: 0.4–0.8)")
    axes[0,0].axvline(0.4, color='r', linestyle='--', label='Veg threshold')
    axes[0,0].axvline(0.8, color='r', linestyle='--')
    
    axes[0,1].hist(np.concatenate(ndbi_all), bins=50, alpha=0.7, edgecolor='k')
    axes[0,1].set_title("NDBI Distribution (Expected: urban 0.0–0.3)")
    axes[0,1].axvline(0.1, color='r', linestyle='--', label='Urban threshold')
    
    axes[1,0].hist(np.concatenate(ndwi_all), bins=50, alpha=0.7, edgecolor='k')
    axes[1,0].set_title("NDWI Distribution (Expected: water 0.3–0.6)")
    axes[1,0].axvline(0.3, color='r', linestyle='--', label='Water threshold')
    
    axes[1,1].hist(np.concatenate(nbr_all), bins=50, alpha=0.7, edgecolor='k')
    axes[1,1].set_title("NBR Distribution (Expected: vegetation 0.4–0.9)")
    axes[1,1].axvline(0.4, color='r', linestyle='--', label='Veg threshold')
    
    plt.tight_layout()
    plt.savefig('results/validation/spectral_indices_distribution.png', dpi=150)
    print("✅ Saved index distributions")
```

**Validation passes if:**
- ✅ All indices stay within [-1, 1]
- ✅ NDVI mean is 0.2–0.8 (indicates vegetation in dataset)
- ✅ NDBI mean is 0.0–0.2 (indicates low urban content, makes sense for wildfire regions)
- ✅ NDWI mean is -0.2–0.3 (indicates mostly non-water)
- ✅ NBR mean is 0.3–0.7 (indicates vegetation)

---

## Part 2: Validate Algorithmic Labels (Correctness)

### 2.1 Visual Inspection of Generated Labels

**Generate labels on 10 random samples:**

```python
def generate_sample_visualizations(num_samples=10):
    """Create visual inspection plots for algorithmic labels"""
    
    for idx in range(num_samples):
        sample = dataset[idx]
        pre_fire = sample['image'][0, :3].numpy()  # RGB for visualization
        
        # Compute spectral indices
        ndvi = compute_ndvi(sample['image'][0])
        ndbi = compute_ndbi(sample['image'][0])
        ndwi = compute_ndwi(sample['image'][0])
        
        # Generate algorithmic labels
        labels = generate_multiclass_labels(ndvi, ndbi, ndwi)
        
        # Visualize
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Indices
        axes[0,0].imshow(ndvi, cmap='RdYlGn')
        axes[0,0].set_title(f"NDVI (sample {idx})")
        axes[0,1].imshow(ndbi, cmap='RdYlBu')
        axes[0,1].set_title("NDBI")
        axes[0,2].imshow(ndwi, cmap='Blues')
        axes[0,2].set_title("NDWI")
        
        # Row 2: RGB + Labels
        rgb_norm = (pre_fire - pre_fire.min()) / (pre_fire.max() - pre_fire.min())
        axes[1,0].imshow(np.transpose(rgb_norm, (1, 2, 0)))
        axes[1,0].set_title("Pre-Fire RGB")
        
        cmap = plt.cm.get_cmap('tab10', 5)  # 5 classes
        axes[1,1].imshow(labels, cmap=cmap, vmin=0, vmax=4)
        axes[1,1].set_title("Generated Labels\n(0=Water, 1=Urban, 2=Dense Veg, 3=Sparse Veg, 4=Other)")
        
        # Ground truth
        axes[1,2].imshow(sample['mask'][0], cmap='RdYlGn_r')
        axes[1,2].set_title("Ground Truth (Burn)")
        
        plt.tight_layout()
        plt.savefig(f'results/validation/sample_{idx:02d}_labels.png', dpi=100)
        plt.close()
    
    print(f"✅ Saved {num_samples} label visualizations")
```

### 2.2 Sanity Checks

**Verify label generation makes sense:**

```python
def sanity_check_labels(num_samples=100):
    """Check that generated labels are plausible"""
    
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    for i in range(num_samples):
        sample = dataset[i]
        labels = generate_multiclass_labels(sample)
        
        for cls in range(5):
            class_counts[cls] += np.sum(labels == cls)
    
    total_pixels = num_samples * 512 * 512
    
    print("\n" + "="*60)
    print("Algorithmic Label Distribution")
    print("="*60)
    print(f"Water (0):            {class_counts[0] / total_pixels * 100:6.2f}%")
    print(f"Urban (1):            {class_counts[1] / total_pixels * 100:6.2f}%")
    print(f"Dense Vegetation (2): {class_counts[2] / total_pixels * 100:6.2f}%")
    print(f"Sparse Vegetation (3):{class_counts[3] / total_pixels * 100:6.2f}%")
    print(f"Other (4):            {class_counts[4] / total_pixels * 100:6.2f}%")
    print("="*60)
    
    # Sanity checks
    water_pct = class_counts[0] / total_pixels
    urban_pct = class_counts[1] / total_pixels
    veg_pct = (class_counts[2] + class_counts[3]) / total_pixels
    
    assert 0 <= water_pct < 0.05, f"Water {water_pct:.2%} seems off"
    assert 0 <= urban_pct < 0.10, f"Urban {urban_pct:.2%} seems high"
    assert 0.40 < veg_pct < 0.95, f"Vegetation {veg_pct:.2%} seems off"
    
    print("✅ Label distribution passes sanity checks")
```

**Validation passes if:**
- ✅ Visual inspection on 10 samples looks plausible
- ✅ Water appears near rivers/lakes
- ✅ Urban appears near visible infrastructure
- ✅ Dense vegetation matches green areas
- ✅ Class distribution is reasonable (40-95% vegetation, <5% water, <10% urban)

---

## Part 3: Validate Multi-Class Pre-Training Effectiveness (Performance)

### 3.1 A/B Test: Model A vs Model B

**Experiment setup:**

```python
def ablation_study():
    """Compare direct binary training vs multi-class pre-training"""
    
    # Model A: Train binary directly
    print("\n" + "="*60)
    print("MODEL A: Direct Binary Training")
    print("="*60)
    model_a = UNet(in_channels=11, out_channels=2)
    optimizer_a = Adam(model_a.parameters(), lr=0.0005)
    criterion_a = BCEDiceLoss()
    
    history_a = train_model(
        model_a, 
        train_loader_binary,  # Binary labels only
        val_loader_binary,
        optimizer_a,
        criterion_a,
        epochs=50,
        name="model_a_binary_only"
    )
    
    iou_a = evaluate_model(model_a, test_loader_binary)
    print(f"Model A - Burn IoU: {iou_a['burned_iou']:.4f}")
    print(f"Model A - Pixel Accuracy: {iou_a['pixel_acc']:.4f}")
    
    # Model B: Multi-class pre-train, then fine-tune
    print("\n" + "="*60)
    print("MODEL B: Multi-Class Pre-Training → Fine-Tune")
    print("="*60)
    
    # Phase 1: Multi-class training
    model_b = UNet(in_channels=11, out_channels=5)  # 5 classes
    optimizer_b1 = Adam(model_b.parameters(), lr=0.0005)
    criterion_b1 = CrossEntropyLoss()
    
    print("\nPhase 1: Multi-class training (5 classes)...")
    history_b1 = train_model(
        model_b,
        train_loader_multiclass,  # Multi-class labels
        val_loader_multiclass,
        optimizer_b1,
        criterion_b1,
        epochs=40,
        name="model_b_multiclass_pretrain"
    )
    
    # Phase 2: Fine-tune for binary
    print("\nPhase 2: Fine-tuning for binary burn detection...")
    
    # Swap classifier layer (5 → 2 classes)
    model_b.classifier = Conv2d(32, 2, kernel_size=1)
    
    optimizer_b2 = Adam(model_b.parameters(), lr=0.0002)  # Lower LR for fine-tuning
    criterion_b2 = BCEDiceLoss()
    
    history_b2 = train_model(
        model_b,
        train_loader_binary,  # Binary labels
        val_loader_binary,
        optimizer_b2,
        criterion_b2,
        epochs=30,  # Fewer epochs on new task
        name="model_b_finetune_binary"
    )
    
    iou_b = evaluate_model(model_b, test_loader_binary)
    print(f"Model B - Burn IoU: {iou_b['burned_iou']:.4f}")
    print(f"Model B - Pixel Accuracy: {iou_b['pixel_acc']:.4f}")
    
    return iou_a, iou_b, history_a, history_b1, history_b2
```

### 3.2 Comparison Metrics

**Metrics to compare:**

```python
def compare_models(iou_a, iou_b, history_a, history_b1, history_b2):
    """Compare Model A and Model B on multiple dimensions"""
    
    print("\n" + "="*70)
    print("COMPARISON: Model A (Binary) vs Model B (Multi-Class Pre-Train)")
    print("="*70)
    
    # 1. Final Performance
    burn_iou_a = iou_a['burned_iou']
    burn_iou_b = iou_b['burned_iou']
    improvement = (burn_iou_b - burn_iou_a) / burn_iou_a * 100
    
    print(f"\n1. FINAL BURN IoU")
    print(f"   Model A (binary only):        {burn_iou_a:.4f}")
    print(f"   Model B (multi-class):        {burn_iou_b:.4f}")
    print(f"   Improvement:                  {improvement:+.1f}%")
    
    if improvement >= 5.0:
        print(f"   ✅ PASS: Model B improves by ≥5%")
    else:
        print(f"   ❌ FAIL: Model B does not improve sufficiently")
    
    # 2. Convergence Speed
    min_loss_a = min(history_a['val_loss'])
    epochs_to_convergence_a = history_a['val_loss'].index(min_loss_a)
    
    min_loss_b = min(history_b2['val_loss'])
    epochs_to_convergence_b = history_b2['val_loss'].index(min_loss_b) + 40
    
    print(f"\n2. CONVERGENCE SPEED")
    print(f"   Model A - Min loss at epoch:  {epochs_to_convergence_a}")
    print(f"   Model B - Min loss at epoch:  {epochs_to_convergence_b}")
    print(f"   Model A total epochs:         {len(history_a['val_loss'])}")
    print(f"   Model B total epochs:         {len(history_b1['val_loss']) + len(history_b2['val_loss'])}")
    
    # 3. Pixel Accuracy
    acc_a = iou_a['pixel_acc']
    acc_b = iou_b['pixel_acc']
    
    print(f"\n3. PIXEL ACCURACY")
    print(f"   Model A:                      {acc_a:.4f}")
    print(f"   Model B:                      {acc_b:.4f}")
    print(f"   Difference:                   {acc_b - acc_a:+.4f}")
    
    # 4. Training Stability
    loss_var_a = np.var(history_a['val_loss'][-10:])
    loss_var_b = np.var(history_b2['val_loss'][-10:])
    
    print(f"\n4. TRAINING STABILITY (loss variance, last 10 epochs)")
    print(f"   Model A:                      {loss_var_a:.6f}")
    print(f"   Model B:                      {loss_var_b:.6f}")
    
    print("\n" + "="*70)
```

### 3.3 Visualization Comparison

**Plot training curves side-by-side:**

```python
def plot_comparison(history_a, history_b1, history_b2):
    """Plot training curves for both models"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Model A losses
    axes[0,0].plot(history_a['train_loss'], label='Train', linewidth=2)
    axes[0,0].plot(history_a['val_loss'], label='Validation', linewidth=2)
    axes[0,0].set_title("Model A: Direct Binary Training")
    axes[0,0].set_ylabel("Loss")
    axes[0,0].set_xlabel("Epoch")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Model B multi-class phase
    axes[0,1].plot(history_b1['train_loss'], label='Train (Multi-class)', linewidth=2)
    axes[0,1].plot(history_b1['val_loss'], label='Val (Multi-class)', linewidth=2)
    axes[0,1].set_title("Model B Phase 1: Multi-Class Pre-Training")
    axes[0,1].set_ylabel("Loss")
    axes[0,1].set_xlabel("Epoch")
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Model B fine-tune phase
    axes[1,0].plot(history_b2['train_loss'], label='Train (Binary)', linewidth=2)
    axes[1,0].plot(history_b2['val_loss'], label='Val (Binary)', linewidth=2)
    axes[1,0].set_title("Model B Phase 2: Fine-Tune on Binary")
    axes[1,0].set_ylabel("Loss")
    axes[1,0].set_xlabel("Epoch")
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # IoU comparison
    axes[1,1].bar(['Model A\n(Binary Only)', 'Model B\n(Multi-Class)'], 
                  [iou_a['burned_iou'], iou_b['burned_iou']], 
                  color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='k', linewidth=2)
    axes[1,1].set_ylabel("Burn IoU")
    axes[1,1].set_title("Final Burn IoU Comparison")
    axes[1,1].set_ylim([0, 1.0])
    for i, (a, b) in enumerate([(iou_a['burned_iou'], 'A'), (iou_b['burned_iou'], 'B')]):
        axes[1,1].text(i, a + 0.02, f'{a:.4f}', ha='center', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/validation/model_comparison.png', dpi=150)
    print("✅ Saved comparison plot")
```

---

## Part 4: Validation Summary Report

### 4.1 Create `VALIDATION_RESULTS.md`

After all experiments, create a report:

```markdown
# Option B Validation Results

**Date**: 2026-06-XX  
**Purpose**: Validate that multi-class pre-training improves burn detection

## Part 1: Spectral Indices Validation ✅
- NDVI range: [-0.23, 0.87] ✅ (expected -1 to 1)
- NDVI mean: 0.42 ✅ (expected 0.2-0.8 for vegetation)
- NDBI range: [-0.15, 0.29] ✅
- NDWI range: [-0.68, 0.45] ✅
- **Conclusion**: Spectral indices computed correctly

See: `results/validation/spectral_indices_distribution.png`

## Part 2: Algorithmic Labels Validation ✅
- Visual inspection on 10 samples: PLAUSIBLE ✅
  - Water labels near rivers: YES
  - Urban labels near infrastructure: YES
  - Vegetation labels match green areas: YES
  - Class distribution reasonable: YES

Distribution across 100 samples:
- Water: 1.2% (expected <5%)
- Urban: 3.1% (expected <10%)
- Dense Vegetation: 52.4%
- Sparse Vegetation: 31.2%
- Other: 12.1%

**Conclusion**: Algorithmic labels are sensible

See: `results/validation/sample_00_labels.png` (+ 9 more)

## Part 3: Multi-Class Pre-Training Effectiveness ⚠️ / ✅

### Model Comparison

| Metric | Model A (Binary) | Model B (Multi-Class) | Improvement |
|--------|-----------------|----------------------|-------------|
| **Burn IoU** | 0.4523 | 0.4761 | +5.3% ✅ |
| Pixel Accuracy | 0.8234 | 0.8301 | +0.8% |
| Convergence Epoch | 38 | 45 (30+15) | Neutral |
| Final Loss | 0.241 | 0.238 | -1.2% |

### Decision: ✅ OPTION B VALIDATES

- Burn IoU improved by 5.3% (threshold: ≥5%) ✅
- Pixel accuracy improved by 0.8%
- Training stable on both paths
- Multi-class pre-training provided useful feature learning

**Recommendation**: Use Option B approach (multi-class pre-train → fine-tune)

See: `results/validation/model_comparison.png`

---

## Conclusion

Option B successfully validates against both algorithmic and performance criteria:

✅ Spectral indices are correct  
✅ Algorithmic labels are plausible  
✅ Multi-class pre-training improves burn detection IoU by 5.3%  

**Final burn model uses Option B approach.**
```

### 4.2 Save All Artifacts

```
results/validation/
├── spectral_indices_distribution.png
├── sample_00_labels.png
├── sample_01_labels.png
├── ...
├── sample_09_labels.png
├── model_comparison.png
└── VALIDATION_RESULTS.md
```

---

## Success Criteria: Decision Table

| Validation Check | Pass Threshold | Status | Decision |
|---|---|---|---|
| **Spectral Indices** | Ranges within [-1, 1] and means sensible | ✅ | Continue |
| **Algorithmic Labels** | Visual inspection looks plausible | ✅ | Continue |
| **Multi-Class Pre-Train** | Burn IoU(B) ≥ Burn IoU(A) + 5% | ✅/❌ | Use/Skip Option B |

**If all pass**: Use Option B  
**If any fail**: Stick with Option A

---

## Timeline

- **Run Part 1-2 validation**: Day 3-4 (during/after multi-class training)
- **Run Part 3 validation**: Day 4-5 (after fine-tuning completes)
- **Write report**: Day 5
- **Include in final documentation**: README and GitHub

---

**Last Updated**: 2026-06-22  
**Author**: Stephen Cerruti  
**Advisor**: Garrison Cottrell
