---
title: PA3 to RETINNA - Semantic Segmentation for Wildfire Detection
subtitle: From Classroom Assignment to Climate Tech
author: RETINNA Educator
date: 2026-06-24
---

# PA3 to RETINNA
## From Classroom to Climate Tech

- **PA3**: UC San Diego semantic segmentation assignment (27 street-scene classes)
- **RETINNA**: Wildfire burn scar detection from Sentinel-2 imagery
- **Connection**: Same FCN principles, different domain
- **Learning goal**: Understand architecture, loss functions, and metrics

---

# FCN Architecture: Encoder-Decoder Pattern

**Input**: Multi-spectral images (11 channels, Sentinel-2)

**Encoder** (Downsampling):
- 5 convolutional blocks, stride=2
- Channel progression: 11 → 32 → 64 → 128 → 256 → 512
- Total downsample: 32× (2^5)

**Decoder** (Upsampling):
- 5 transposed convolution blocks, stride=2
- Reverses encoder: 512 → 256 → 128 → 64 → 32 → 2 classes

**Key**: Batch normalization after every layer; ReLU between layers

---

# Skip Connections: U-Net Improves FCN

**FCN (PA3)**: Information flows strictly encoder → decoder
- Loses fine spatial details at bottleneck
- Good for large features; misses small structures

**U-Net (RETINNA)**: Skip connections preserve detail
- Copy encoder features to matching decoder scales
- Better boundary preservation
- Higher recall on small burn scars

**Trade-off**: ~3× more parameters; better accuracy justifies it

---

# Handling Class Imbalance

**Problem**: ~90% unburned, ~10% burned (severe imbalance)

**PA3 approach**: CrossEntropy works for 27 balanced classes

**Our approach**: Hybrid BCE + Dice loss
- **BCE**: Pixel-level classification loss
- **Dice**: Region-level overlap metric
- **pos_weight**: Penalize false negatives (missed burns)

**Result**: Model learns subtle burn signals without predicting "unburned" everywhere

---

# Validation Results: False Negatives Heatmap

![False Negative Heatmap](./visualizations/false_negative_heatmap.png)

**Metrics Summary**:
- **IoU (Burned Class)**: 0.88
- **Precision**: 1.00 (high confidence when predicting burn)
- **Recall**: 0.88 (catches most actual burns)
- **False Negative Rate**: 0.12 (12% of burns missed)

**Key finding**: Model detects bright burns well; lightweight slope fires remain challenging

---

# Learning Objectives Achieved

✓ **FCN Architecture**: Encoder/decoder symmetry and spatial reconstruction

✓ **Skip Connections**: U-Net improvements for detail preservation

✓ **Loss Functions**: Class imbalance handling through hybrid losses

✓ **Evaluation Metrics**: IoU, precision, recall and their interpretations

✓ **Hyperparameter Tuning**: Batch size, learning rate, loss weighting

✓ **Real-World Application**: Translating PA3 concepts to satellite imagery domain
