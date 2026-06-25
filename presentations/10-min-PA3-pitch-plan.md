# 10-Minute PA3 Presentation Plan

## Target
- **Duration:** 10 minutes
- **Slides:** 5-6 slides max
- **Words per slide:** <50 words
- **Design principle:** One idea per slide
- **Academic focus:** Learning objectives, not flashy results

## Slide Breakdown

### Slide 1: Context & Motivation
**Title:** From PA3 Assignment to Real-World Wildfire Detection

**Content:**
- PA3: Semantic segmentation on urban street scenes (27 classes)
- RETINNA adaptation: Binary burn scar detection from Sentinel-2
- Why it matters: Wildfire monitoring, climate resilience
- Common foundation: FCN architecture, loss functions, metrics

**Elements:** Title, 3-4 bullet points
**Word count:** ~40 words

---

### Slide 2: FCN Architecture Template
**Title:** Fully Convolutional Network: Encoder-Decoder Pattern

**Content:**
- Input: Multi-spectral images (11 channels for Sentinel-2)
- Encoder: 5 conv blocks, stride=2 downsampling (32× total)
- Decoder: 5 deconv blocks, stride=2 upsampling (back to resolution)
- Output: 2-class predictions (burned vs. unburned)
- Key detail: Batch normalization after each layer

**Elements:** Architecture diagram (ASCII or simple box diagram), key layer counts
**Word count:** ~45 words

---

### Slide 3: U-Net vs. FCN Comparison
**Title:** Skip Connections: FCN vs. U-Net

**Content:**
- FCN (PA3): Information flows strictly down then up (encoder→decoder)
- U-Net (RETINNA): Skip connections preserve fine details across scales
- Impact: Better boundary preservation, higher IoU on small burn scars
- Trade-off: More parameters, slightly longer inference

**Elements:** Side-by-side architecture diagram, visual highlighting skip connections
**Word count:** ~40 words

---

### Slide 4: Handling Class Imbalance
**Title:** Loss Function & Training Strategy for Severe Imbalance

**Content:**
- Problem: ~90% unburned, ~10% burned (severe imbalance)
- PA3 uses CrossEntropy (27 balanced classes); RETINNA needs hybrid approach
- Solution: BCE (pixel-level) + Dice loss (area-level)
- Tuning: pos_weight parameter penalizes false negatives
- Result: Model learns to detect subtle burn signals

**Elements:** Loss formula reference (optional), imbalance visualization
**Word count:** ~48 words

---

### Slide 5: Validation Results & False Negatives
**Title:** Model Performance: Where It Succeeds & Fails

**Content:**
- Metrics: IoU (Intersection over Union) for burned class
- Key finding: False negatives (missed burns) remain a challenge
- Visualization: Pixel-level heatmap of false negatives
- Metrics table: IoU, Precision, Recall, False Negative Rate
- Learning: Why lightweight burns are hard to detect

**Elements:** Heatmap image, metrics table, 1-2 bullet points
**Word count:** ~45 words

---

### Slide 6: Learning Objectives Achieved
**Title:** What You've Learned (Technical Competencies)

**Content:**
1. FCN architecture: encoder/decoder symmetry
2. Skip connections: U-Net improvements over basic FCN
3. Loss functions: Handling class imbalance in segmentation
4. Evaluation metrics: IoU, precision, recall for binary classification
5. Hyperparameter tuning: Batch size, learning rate, loss weighting
6. From academic assignment to real-world application

**Elements:** Checklist or numbered list, checkmarks
**Word count:** ~50 words

---

## Visualization Requirements

### For Slide 5 (False Negatives Heatmap)
Extract from training notebook:
1. Model predictions on validation set
2. Ground truth mask
3. Compute false negatives: predicted=0, actual=1 (missed burns)
4. Create pixel-level heatmap (red = false negative, green = correct detection, gray = unburned)
5. Include metrics table below: IoU, Precision, Recall, FN_Rate

## Notes for Generator
- Use clear, minimal typography
- Diagrams should use only black/white/red (accessibility)
- Include speaker notes with technical details
- Emphasize learning objectives (assignment context)
- Avoid marketing language; stay academic
