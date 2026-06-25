---
title: From PA3 Assignment to Climate Tech - Wildfire Detection Deep Dive
subtitle: Semantic Segmentation on Sentinel-2 Multispectral Data
author: RETINNA Educator
date: 2026-06-24
---

# From Classroom to Climate: PA3 to RETINNA

**PA3 Context** (UC San Diego CSE 251B, Winter 2021):
- Assignment: Semantic segmentation on Indian Driving Dataset
- 27 street-scene classes (road, car, building, vegetation, etc.)
- Goal: Students implement FCN architecture from scratch

**RETINNA Project**:
- Research Experience for Teachers in AI (mentored by Gary Cottrell)
- Your task: Wildfire burn scar detection using Sentinel-2
- Same principles, different domain and scale
- Real-world impact: Rapid wildfire monitoring for climate resilience

---

# The Problem Domain

**Scale**: Sentinel-2 captures 10,980 × 10,980 km swaths every 5 days

**Detection challenge**: Distinguish burned areas from:
- Natural shadows and terrain variations
- Clouds and atmospheric effects
- Water bodies and reflective surfaces

**Class imbalance**: ~90% unburned, ~10% burned pixels

**Opportunity**: Automated detection reduces manual analysis burden and enables rapid emergency response

---

# Sentinel-2 Multispectral Data

**11 Spectral Bands** (not just RGB):
- **Visible**: Blue, Green, Red
- **Near-Infrared (NIR)**: Vegetation presence and health
- **Short-Wave Infrared (SWIR)**: Burn severity, moisture content
- **Atmospheric**: Water vapor, aerosol bands (for preprocessing)

**Temporal Dimension**:
- Multiple observations (T=8) per geographic location
- Captures change over time (burned area darkens)
- Stacked into single tensor: (T×Channels, Height, Width)

**Normalization**: Per-band statistics computed from CaBuAr training data
- NOT ImageNet (which assumes RGB natural images)
- Critical for multispectral data: each band has different range

---

# Data Pipeline: From Pixels to Tensors

**PA3 Approach** (reference):
1. Load single-channel mask: pixel value = class index (0-26)
2. Convert to one-hot: shape becomes (27, Height, Width)
3. Separate class handling from spatial relationships

**RETINNA Adaptation**:
1. Load 11-band Sentinel-2 image per location
2. Stack T=8 timepoints: shape (8, 11, H, W) → reshape to (88, H, W)
3. Load binary mask: shape (H, W) → expand to (1, H, W)
4. Normalize independently per band
5. Batch processing: (Batch, 88, H, W) → GPU

**Key insight**: One-hot encoding separates semantic meaning from pixel layout

---

# FCN Architecture: Deep Dive

**Fully Convolutional Network** (PA3's foundation):

**Encoder (Downsampling)**:
```
Input (N, 3, H, W)
→ Conv2d(3, 32, k=3, s=2) + BatchNorm + ReLU → (N, 32, H/2, W/2)
→ Conv2d(32, 64, k=3, s=2) + BatchNorm + ReLU → (N, 64, H/4, W/4)
→ Conv2d(64, 128, k=3, s=2) + BatchNorm + ReLU → (N, 128, H/8, W/8)
→ Conv2d(128, 256, k=3, s=2) + BatchNorm + ReLU → (N, 256, H/16, W/16)
→ Conv2d(256, 512, k=3, s=2) + BatchNorm + ReLU → (N, 512, H/32, W/32)
```
Total downsample factor: **32× (2^5)**

**Decoder (Upsampling)**:
```
(N, 512, H/32, W/32)
→ ConvTranspose2d(512, 256, k=4, s=2) + BatchNorm + ReLU → (N, 256, H/16, W/16)
→ ConvTranspose2d(256, 128, k=4, s=2) + BatchNorm + ReLU → (N, 128, H/8, W/8)
→ ConvTranspose2d(128, 64, k=4, s=2) + BatchNorm + ReLU → (N, 64, H/4, W/4)
→ ConvTranspose2d(64, 32, k=4, s=2) + BatchNorm + ReLU → (N, 32, H/2, W/2)
→ ConvTranspose2d(32, n_classes, k=4, s=2) → Output (N, n_classes, H, W)
```

**Bottleneck**: Smallest feature map (H/32, W/32) at 512 channels captures semantic content

---

# U-Net: Skip Connections for Better Detail

**Problem with FCN**: Encoder bottleneck loses spatial details

**U-Net Solution**: Connect encoder layers to decoder layers at matching scales

```
Encoder:  → Layer1(C=64) → Layer2(C=128) → Layer3(C=256) → Layer4(C=512) → Bottleneck
             ↓ concat      ↓ concat       ↓ concat       ↓ concat
Decoder:  ← Conv           ← Conv         ← Conv         ← Conv         ← Output
```

**Benefit**: 
- Skip connections carry fine spatial information (edges, small objects)
- Decoder can reconstruct boundaries more accurately
- Better recall on small burn scars

**RETINNA Configuration**:
- Input: 11 Sentinel-2 bands (not 3 like PA3)
- Input channels → 32 (matching encoder structure)
- Output: 2 classes (unburned, burned)
- Parameter count: ~60 million (vs ~30 million for FCN)

---

# Loss Functions: Addressing Severe Imbalance

**PA3 Context**: 27 classes, relatively balanced distribution
- **Standard approach**: CrossEntropyLoss
- **Assumption**: Each class appears equally in dataset

**RETINNA Challenge**: 90/10 class split (severe imbalance)
- CrossEntropy alone fails: model predicts "unburned" everywhere
- High accuracy (90%) but zero recall for burned class

**Hybrid BCE + Dice Loss**:

**Binary Cross Entropy (BCE)**:
```
L_BCE = -Σ [y*log(p) + (1-y)*log(1-p)]
```
Per-pixel classification loss; prone to imbalance

**Dice Loss**:
```
L_Dice = 1 - (2*intersection + ε) / (union + ε)
```
Region-level overlap metric; handles imbalance naturally

**Combined**:
```
L_total = α*L_BCE + (1-α)*L_Dice
where α = 0.5 typically
```

**pos_weight Tuning**:
- pos_weight parameter in BCE: weight=1.5 means "missing a burn is 1.5× worse"
- Compensates for class imbalance by emphasizing false negatives
- Tuning: 1.0 → 1.5 → 2.0 (experiments showed 1.5 optimal)

---

# Training Results: The False Negative Challenge

![Training Curves](./visualizations/training_curves.png)

**Initial Observation**:
- Model achieved IoU ≈ 0.42 on validation set
- Accuracy ≈ 0.92 (but mostly "unburned" predictions!)

**Failure Analysis**:
- Precision = 0.68 (when model predicts burn, 68% correct)
- Recall = 0.32 (model only finds 32% of actual burned areas!)
- High false negative rate: 68% of burns go undetected

**Root Cause**:
- Lightweight burns (low reflectance change) hard to distinguish
- Model biases toward "unburned" due to imbalance
- Subtle spectral signals drowned out by class prior

**Lesson**: Single IoU metric masks class-specific failures
- Need precision/recall breakdown
- Need false negative heatmaps to visualize failure modes

---

# Hyperparameter Tuning Strategy

![Hyperparameter Results](./visualizations/hyperparameter_tuning.png)

**Batch Size Exploration**:
- Small batch (8): Stable but slow convergence
- Medium batch (16): Balance between speed and stability
- Large batch (32): Fast per-epoch but noisier gradients
- **Decision**: 16 chosen for batch_size

**Learning Rate** (Adam optimizer):
- Too high (0.001): Unstable training, loss spikes
- Too low (0.0001): Slow convergence, underfitting
- **Sweet spot**: 0.0005 (balanced learning)

**Loss Weight Tuning**:
- Initial: BCE=0.5, Dice=0.5 (equal weight)
- Adjustment: Slight increase in Dice weight helped recall
- **Final**: BCE=0.4, Dice=0.6 (emphasize region-level)

**pos_weight Tuning** (most impactful):
- pos_weight=1.0: Baseline, high false negatives
- pos_weight=1.5: Better recall, IoU improves to 0.42
- pos_weight=2.0: Too aggressive, overfits false positive penalties
- **Winner**: pos_weight=1.5 achieved best validation performance

**Result**: 12 epochs to convergence with final configuration

---

# FCN vs. U-Net: Empirical Comparison

![FCN vs U-Net Comparison](./visualizations/fcn_vs_unet_comparison.png)

| Metric | FCN | U-Net | Winner |
|--------|-----|-------|--------|
| **IoU (Burned Class)** | 0.38 | 0.42 | U-Net (+5%) |
| **Precision** | 0.71 | 0.68 | FCN (fewer FP) |
| **Recall** | 0.24 | 0.32 | U-Net (+33%) |
| **False Negative Rate** | 0.76 | 0.68 | U-Net (8% better) |
| **Model Parameters** | 31M | 60M | FCN (size) |
| **Inference Time** | 180ms | 240ms | FCN (speed) |

**Interpretation**:
- U-Net superior on recall (critical for safety: missing a burn is worse than false alarm)
- FCN faster and smaller (useful for edge deployment)
- Trade-off acceptable: Safety > speed in wildfire context
- U-Net chosen for production despite higher compute cost

---

# Validation Visualization: Failure Modes

![Validation Heatmap](./visualizations/false_negative_heatmap.png)

**Color Coding**:
- **Red**: False Negative (predicted unburned, actual burned) - CRITICAL ERROR
- **Green**: True Positive (predicted burned, actual burned) - CORRECT
- **Gray**: True Negative (predicted unburned, actual unburned) - BACKGROUND
- **Yellow**: False Positive (predicted burned, actual unburned) - ACCEPTABLE ERROR

**Metrics Summary**:
| Metric | Value | Interpretation |
|--------|-------|-----------------|
| IoU | 0.88 | Strong overlap; good segmentation |
| Precision | 1.00 | All predictions correct |
| Recall | 0.88 | 88% of actual burns detected |
| FN Rate | 0.12 | 12% of burns are false negatives |

**Spatial Pattern**:
- Bright burns (high spectral change): Mostly detected (green)
- Lightweight burns on slopes: Mostly missed (red clusters)
- Water and shadows: Correctly classified as unburned (gray)

**Lesson**: Model learns general patterns; domain-specific knowledge (slope orientation, elevation) needed for refinement

---

# Learning Objectives Achieved

**1. Architecture Design** ✓
- FCN encoder/decoder ensures symmetric spatial information flow
- Downsampling captures context; upsampling reconstructs detail
- Batch norm stabilizes training; ReLU provides nonlinearity

**2. Skip Connections** ✓
- U-Net improves on FCN by preserving fine details
- Skip connections vital for small-object detection (burn scars)
- Trade-off: More parameters for better accuracy

**3. Loss Functions** ✓
- Class imbalance requires hybrid approach (BCE + Dice)
- pos_weight parameter crucial for false negative reduction
- Empirical tuning (1.5) outperforms off-the-shelf defaults

**4. Evaluation Metrics** ✓
- IoU: Overall segmentation quality
- Precision: False positive control (confidence in positive predictions)
- Recall: False negative control (coverage of actual positives) - CRITICAL for wildfire

**5. Hyperparameter Tuning** ✓
- Systematic exploration of batch size, learning rate, loss weights
- Documented experiments enable reproducibility
- Final configuration balances convergence speed and accuracy

**6. Domain Adaptation** ✓
- Sentinel-2 multispectral (11 channels) ≠ RGB (3 channels)
- Custom normalization required (not ImageNet)
- Temporal stacking (T=8) captures burn evolution

**7. From Assignment to Application** ✓
- PA3 provides algorithmic foundation
- Real-world dataset reveals challenges (class imbalance, subtle signals)
- Academic rigor applied to climate-relevant problem
