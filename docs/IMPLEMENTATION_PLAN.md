# Implementation Plan: Wildfire Burn Scar Segmentation

**Status**: Active  
**Timeline**: 1 week (5 days, half-day commitment = ~20 hours) + beyond  
**Primary Goal**: Implement Option A (binary burn detection on Sentinel-2)  
**Stretch Goal 1**: Option B (multi-class pre-training on Sentinel-2) if time allows  
**Stretch Goal 2**: Option C (cross-satellite transfer to Landsat 8) if time/interest allows

---

## Quick Summary: Three Options

| Option | What | Time | Difficulty | Value |
|--------|------|------|------------|-------|
| **A** (Primary) | Binary burn detection on Sentinel-2 | 9-15h | Medium | Core skill: U-Net + training |
| **B** (Stretch) | Multi-class pre-training → fine-tune | +5-9h | Medium-High | Improvement: Better burn detection |
| **C** (Advanced) | Transfer to Landsat 8 (different satellite) | +5-7h | Medium-High | Generalization: Satellite-agnostic model |

**Tip**: A is required. After A works, you'll see what Gary wants. B and C are independent paths (not sequential).

---

## Option A: Binary Burn Detection (Primary Goal)

Build a custom U-Net for binary segmentation (unburned vs. burned) on CaBuAr dataset.

**Output**: 2 classes (unburned=0, burned=1)

### Deliverables

#### 1. `src/unet.py` — U-Net Architecture
- Custom U-Net implementation from scratch
- Input: 11 channels (Sentinel-2 multispectral)
- Output: 2 classes (unburned, burned)
- Skip connections (encoder → decoder)
- Batch normalization, ReLU activations
- **Time**: 2-3 hours
- **Maps to Issue**: [#10 Model Architecture](https://github.com/scerruti/RETINNA/issues/10)

#### 2. `src/dataset.py` — CaBuAr Data Pipeline
- Load pre/post-fire image pairs from CaBuAr
- Compute per-band normalization statistics (Sentinel-2, not ImageNet)
- One-hot encoding for 2-class labels
- Data augmentation (rotation, flip, crop)
- Train/val/test split handling
- **Time**: 1-2 hours
- **Maps to Issues**: [#3 Data Loading](https://github.com/scerruti/RETINNA/issues/3), [#4 Exploratory Analysis](https://github.com/scerruti/RETINNA/issues/4), [#8 Data Normalization](https://github.com/scerruti/RETINNA/issues/8)

#### 3. `train.py` — Training Script
- Load model and dataset
- Loss function: BCE + Dice (weighted for class imbalance)
- Optimizer: Adam (configurable learning rate)
- Per-epoch validation with IoU tracking
- Model checkpointing (save best model)
- Early stopping (monitor validation IoU)
- Logging (loss, metrics per epoch)
- **Time**: 2-3 hours
- **Maps to Issues**: [#11 Training Script](https://github.com/scerruti/RETINNA/issues/11), [#12 Baseline Evaluation](https://github.com/scerruti/RETINNA/issues/12)

#### 4. `src/metrics.py` — Evaluation Metrics
- IoU (Intersection over Union) for burned class (class 1)
- Pixel accuracy
- Per-class metrics logging
- **Time**: 1 hour
- **Maps to Issue**: [#12 Baseline Evaluation](https://github.com/scerruti/RETINNA/issues/12)

#### 5. `evaluate.py` — Testing & Visualization
- Load trained model
- Run on test set
- Compute final IoU and pixel accuracy
- Generate visualizations:
  - Pre-fire RGB
  - Post-fire RGB
  - Ground truth mask
  - Predicted mask
  - Overlay comparison
- Save results/plots
- **Time**: 1 hour
- **Maps to Issue**: [#19 Inference API Build](https://github.com/scerruti/RETINNA/issues/19)

#### 6. `src/hyperparameters.py` or training config
- Batch size: 8-32 (test range, start at 16)
- Learning rate: 0.0001-0.001 (test range, start at 0.0005)
- Epochs: 30-100 (with early stopping)
- Loss weights: BCE + α*Dice (test α = 0.5, 0.7, 1.0)
- **Time**: Integrated into training
- **Maps to Issues**: [#14 Hyperparameter Tuning](https://github.com/scerruti/RETINNA/issues/14), [#15 Loss Optimization](https://github.com/scerruti/RETINNA/issues/15)

#### 7. Results & Documentation
- Training curves (loss, IoU over epochs)
- Final test metrics (IoU, pixel accuracy)
- Inference examples (3-4 samples with visualizations)
- README documenting hyperparameters, results, how to run
- **Time**: 1-2 hours
- **Maps to Issue**: [#20 Final Documentation](https://github.com/scerruti/RETINNA/issues/20)

### Time Budget (Option A)

| Task | Time | Status |
|------|------|--------|
| U-Net architecture | 2-3h | |
| Data pipeline | 1-2h | |
| Training script | 2-3h | |
| Metrics | 1h | |
| Evaluation & visualization | 1h | |
| Hyperparameter tuning | 1-2h | |
| Documentation & results | 1-2h | |
| **Total Option A** | **9-15h** | |
| Buffer | ~5-11h | |

---

## Option C: Cross-Satellite Transfer Learning to Landsat 8 (Advanced Stretch Goal)

**IF both Option A works well AND you want to demonstrate satellite generalization**, implement cross-satellite transfer.

**Approach:**
Take your trained Sentinel-2 burn detection model and transfer it to Landsat 8 imagery, showing that burn detection generalizes across different satellites.

**Output**: Burn detection model works on both Sentinel-2 (10m) and Landsat 8 (30m) imagery

### Why This Matters

- Demonstrates practical satellite data knowledge
- Shows transfer learning across domains (different satellites, different resolutions)
- Landsat has advantages: longer time series, global coverage, free access
- Real-world applicability: many regions only have Landsat data
- Unlike Option B (same satellite, different approach), Option C shows **generalization** (key ML concept)

### Deliverables

#### 1. `src/landsat_loader.py` — Landsat 8 Dataset Pipeline
- Load Landsat 8 data from Google Earth Engine or USGS
- Map pre/post-fire Landsat image pairs
- Handle band selection (Landsat has 11 bands, need NIR + Red for NDVI)
- Normalize to Sentinel-2 scale (for transfer) OR use Landsat-specific normalization
- **Time**: 2-3 hours (more complex: different data source, different format)
- **Challenge**: Finding Landsat pairs with known burn boundaries
  - Option A: Use Sentinel-2 predictions as pseudo-labels for Landsat
  - Option B: Find USGS/NASA burn maps that cover Landsat footprints

#### 2. `transfer_to_landsat.py` — Transfer Learning Script
- Load Sentinel-2 trained model (from Option A)
- Adapt for Landsat input (Landsat has 11 bands, not 12)
  - Option 1: Use same model, ignore 1 band
  - Option 2: Retrain input layer for 11 channels
- Fine-tune on Landsat data with frozen encoder, trainable decoder
- Monitor burn IoU on Landsat validation set
- **Time**: 1-2 hours
- **Key question**: Do spectral signatures transfer between satellites?

#### 3. `evaluate_landsat.py` — Cross-Satellite Evaluation
- Run Sentinel-2 model predictions on Landsat test data (if compatible)
- Generate burn maps at 30m resolution (vs Sentinel-2's 10m)
- Compare: Sentinel-2 results vs Landsat results on same fires
- Visualize: Side-by-side comparison at different resolutions
- **Time**: 1 hour

#### 4. Results & Documentation
- Comparison report: "Burn Detection Across Sentinel-2 and Landsat 8"
- Visualizations: Burn maps at 10m vs 30m resolution
- Discussion: Resolution trade-offs, generalization insights
- **Time**: 1 hour

### Time Budget (Option C)

| Task | Time |
|------|------|
| Landsat data loader | 2-3h |
| Transfer learning script | 1-2h |
| Cross-satellite evaluation | 1h |
| Results & documentation | 1h |
| **Total Option C** | **5-7h** |

**GPU training time** (parallel):
- Fine-tuning on Landsat: 30min-2h

### Technical Challenges (Option C)

1. **Data Alignment**
   - Sentinel-2 and Landsat footprints differ
   - Different spatial resolution (10m vs 30m)
   - May need to regrid or interpolate
   - **Solution**: Use geospatial tools (rasterio, GDAL) or Google Earth Engine

2. **Spectral Differences**
   - Different sensors → different spectral responses
   - Band naming: Sentinel-2 B8 ≠ Landsat 8 B5 (both NIR but different wavelengths)
   - May need spectral harmonization
   - **Solution**: Use published Sentinel-2 ↔ Landsat 8 band mappings

3. **Label Scarcity**
   - Landsat doesn't have pre-labeled burn data like CaBuAr
   - Options:
     - Use Sentinel-2 model predictions as pseudo-labels (transfer labels)
     - Find USGS/NASA Monitoring Trends in Burn Severity (MTBS) data
     - Manually label a few test samples
   - **Most practical**: Use Sentinel-2 predictions as weak labels

4. **Resolution Trade-off**
   - Landsat 30m is coarser → burn boundaries less precise
   - Expect lower IoU than Sentinel-2
   - Is this acceptable? Depends on use case.

### Approach: Step-by-Step

```
Step 1: Acquire Landsat Data
├─ Use Google Earth Engine or USGS EROS
├─ Find Landsat image pairs (pre/post fire)
├─ For best results: overlay with known burn boundaries (MTBS, CalFire)
└─ Time: 1-2h

Step 2: Generate Labels for Landsat
├─ Option A: Run Sentinel-2 model on Landsat
│  └─ Use predicted burn maps as pseudo-labels
├─ Option B: Use published burn maps (MTBS)
│  └─ More authoritative, but may not align with Landsat footprints
└─ Time: 1h

Step 3: Transfer Learning
├─ Load Sentinel-2 model weights
├─ Adapt input layer for Landsat channels (11 vs 12)
├─ Freeze encoder, train decoder on Landsat data
├─ Monitor: Does burn IoU improve or degrade?
└─ Time: 2h (implementation + training)

Step 4: Evaluate & Compare
├─ Run both models on overlapping Landsat/Sentinel-2 areas
├─ Compare: Sentinel-2 (10m) vs Landsat (30m) burn detection
├─ Visualize at different scales
├─ Document trade-offs
└─ Time: 1-2h
```

### Success Criteria (Option C)

**Minimum (just works)**:
- ✅ Landsat data loads and model runs forward pass
- ✅ Transfer learning script trains without errors
- ✅ Can generate burn maps on Landsat data
- ✅ Visualizations show reasonable results

**Good (shows value)**:
- ✅ Landsat burn IoU > 0.3 (acceptable given 30m resolution)
- ✅ Model generalizes without catastrophic forgetting
- ✅ Comparison shows trade-offs clearly (resolution vs availability)
- ✅ Documentation explains when to use Landsat vs Sentinel-2

**Excellent (publishable quality)**:
- ✅ Landsat burn IoU competitive with Sentinel-2 (within 10%)
- ✅ Spectral harmonization successful
- ✅ Successfully transfer labels from Sentinel-2 to Landsat
- ✅ Discussion of practical implications (which satellite for which use case)

### When to Start Option C

**Prerequisites:**
- [ ] Option A is fully working (trained model, inference tested)
- [ ] You understand Sentinel-2 results well
- [ ] You have > 5-7 hours remaining
- [ ] You're comfortable with geospatial data tools (rasterio, GDAL, or Google Earth Engine)

**If any prerequisite fails**, skip Option C. Not required, only if time/interest.

### Option C Independence

**Important**: Option C is **independent of Option B**
- Can do A → C (without B's multi-class pre-training)
- C tests **generalization across satellites**
- B tests **improvement via multi-task learning**
- Different experiments, different value propositions

**Decision tree:**
```
Complete Option A ✅
├─ Have time? → Try Option B (improve Sentinel-2)
├─ Have time? → Try Option C (generalize to Landsat)
└─ Have time? → Try both B and C (but likely too much)
```

---

## Option B: Multi-Class Pre-Training (Stretch Goal)

**IF time allows after Option A is working**, implement multi-class pre-training.

**Approach:**
1. Compute spectral indices from Sentinel-2 bands
2. Generate multi-class labels algorithmically (land cover classification)
3. Train U-Net on multi-class task
4. Fine-tune on binary burn detection

**Output**: 5 classes (or more)
- 0: Water
- 1: Urban/Built-up
- 2: Dense Vegetation (unburned)
- 3: Sparse Vegetation (unburned)
- 4: Burned

### Additional Deliverables

#### 1. `src/spectral_indices.py` — Index Computation
- NDVI (Normalized Difference Vegetation Index)
- NDBI (Normalized Difference Built-in Index)
- NDWI (Normalized Difference Water Index)
- NBR (Normalized Burn Ratio)
- Compute for pre-fire image only
- **Time**: 1-2 hours

#### 2. `src/label_generator.py` — Algorithmic Label Generation
- Read spectral indices
- Apply thresholds to create class assignments:
  - NDWI > threshold → Water
  - NDBI > threshold → Urban
  - NDVI > high_threshold → Dense Vegetation
  - NDVI > low_threshold → Sparse Vegetation
  - else → Other
- Save as multi-class labels (5-class output)
- **Time**: 1-2 hours

#### 3. `train_multiclass.py` — Multi-Class Training
- Load model and multi-class dataset
- Loss function: CrossEntropyLoss (balanced classes or weighted)
- Same training loop as Option A
- Checkpoint best model on multi-class validation IoU
- **Time**: 0.5-1 hour (reuse Option A training script)

#### 4. `finetune_burns.py` — Fine-Tuning on Binary Task
- Load multi-class trained model
- Option A: Continue training on binary labels with lower learning rate
- Option B: Swap final classification layer (5 → 2 channels) and retrain
- Monitor burn IoU (class 1) specifically
- **Time**: 0.5-1 hour (reuse Option A training script)

#### 5. Dual Evaluation
- Metrics on multi-class task (per-class accuracy, macro IoU)
- Metrics on binary burn task (burn IoU, pixel accuracy)
- Visualizations for both tasks
- **Time**: 1 hour

### Time Budget (Option B Additions)

| Task | Time |
|------|------|
| Spectral indices computation | 1-2h |
| Label generation algorithm | 1-2h |
| Multi-class training | 0.5-1h |
| Fine-tuning on burns | 0.5-1h |
| Dual evaluation | 1h |
| Testing/debugging | 1-2h |
| **Total Option B additions** | **5-9h** |
| **Total Option A + B** | **14-24h** |

**GPU training time** (runs in parallel, doesn't count):
- Multi-class training: 30min-2h
- Fine-tuning: 30min-2h

### Decision Criteria: When to Start Option B

Start Option B **only if:**
- [ ] Option A is fully working (model trains, validation runs, no crashes)
- [ ] You have > 8 hours remaining in the week
- [ ] You understand the spectral index thresholds (research NDVI ranges for vegetation, etc.)
- [ ] You're confident in the additional complexity

**If any of these are false**, skip Option B and focus on polishing Option A.

---

## Sprint Mapping

### Day 1 (Dataset & EDA)
- [ ] Load CaBuAr via TorchGeo
- [ ] Explore dataset structure
- [ ] Compute statistics (burned%, class balance)
- [ ] Visualize sample tiles
- **Issues**: [#3](https://github.com/scerruti/RETINNA/issues/3), [#4](https://github.com/scerruti/RETINNA/issues/4)
- **Time**: 2h

### Day 2 (Data Pipeline & Preprocessing)
- [ ] Implement dataset.py (loader, augmentation)
- [ ] Compute Sentinel-2 normalization statistics
- [ ] Handle bi-temporal data structure
- **Issues**: [#6](https://github.com/scerruti/RETINNA/issues/6), [#7](https://github.com/scerruti/RETINNA/issues/7), [#8](https://github.com/scerruti/RETINNA/issues/8)
- **Time**: 3h

### Day 3 (Model & Training)
- [ ] Implement unet.py
- [ ] Implement train.py
- [ ] Run baseline training (few epochs on sample data)
- [ ] Implement metrics.py
- **Issues**: [#10](https://github.com/scerruti/RETINNA/issues/10), [#11](https://github.com/scerruti/RETINNA/issues/11), [#12](https://github.com/scerruti/RETINNA/issues/12)
- **Time**: 3h (+ GPU training in parallel)

### Day 4 (Optimization & Hyperparameters)
- [ ] Hyperparameter sweeps (batch size, learning rate)
- [ ] Loss function tuning (BCE + Dice weights)
- [ ] Full training run
- **Issues**: [#14](https://github.com/scerruti/RETINNA/issues/14), [#15](https://github.com/scerruti/RETINNA/issues/15), [#16](https://github.com/scerruti/RETINNA/issues/16)
- **Time**: 3h (+ GPU training)

### Day 5 (Evaluation & Documentation)
- [ ] Run evaluate.py on test set
- [ ] Generate visualizations
- [ ] Write documentation
- [ ] Final testing
- **Issues**: [#18](https://github.com/scerruti/RETINNA/issues/18), [#19](https://github.com/scerruti/RETINNA/issues/19), [#20](https://github.com/scerruti/RETINNA/issues/20)
- **Time**: 3h

**If Option B is happening**, overlap multi-class training on Days 3-4 while working on other tasks.

---

## Success Criteria

### Option A (Minimum)
- [ ] U-Net loads and runs forward pass
- [ ] Training completes without errors
- [ ] Loss decreases over epochs
- [ ] Validation IoU > 0.05 (shows model learning)
- [ ] Can save/load trained model
- [ ] Generate inference visualizations
- [ ] README documents hyperparameters and results

### Option B (If Attempted)
- [ ] Multi-class labels generated correctly
- [ ] Multi-class training works
- [ ] Fine-tuning on burns works
- [ ] Both metrics reported (multi-class accuracy + burn IoU)
- [ ] Visualizations show both tasks

---

## Notes

- **PA3 reference**: This plan implements PA3's objectives (architecture, training loop, metrics) on your actual wildfire problem
- **GPU time**: Use Google Colab for training (GPU-accelerated)
- **Local work**: Implement and test locally; push to GitHub; train on Colab
- **If stuck**: Reference PA3_SUMMARY.md in docs/ for technical guidance on each component
- **Time estimate includes**: implementation, testing, debugging, documentation. **Does NOT include** GPU training (runs in parallel)

---

**Last Updated**: 2026-06-22  
**Author**: Stephen Cerruti  
**Advisor**: Garrison Cottrell
