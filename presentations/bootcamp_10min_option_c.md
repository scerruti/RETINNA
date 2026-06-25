---
theme: default
highlighter: shiki
lineNumbers: false
drawings:
  persist: false
transition: slide
title: "Burn Scar Detection: From Failure to Transfer Learning"
mdc: true
---

# Burn Scar Detection
## From Failure to Transfer Learning

A journey through data quality, spectral analysis, and architectural design

<style>
  h1 { font-size: 3.5em; margin-bottom: 0.2em; }
  h2 { font-size: 2.2em; color: #666; }
  .subtitle { font-size: 1.2em; color: #999; line-height: 1.8; }
</style>

---
layout: two-cols
---

# Phase I: U-Net Architecture

::left::

### 24-Channel Model
- **Input**: 2 timesteps × 12 Sentinel-2 bands
- **Spatial**: 512×512 pixels
- **Architecture**: U-Net with skip connections
- **Output**: 2 classes (burned/unburned)
- **Parameters**: 31.1M

### Why U-Net?
- **Skip connections** preserve boundary detail
- Encoder downsamples → Bottleneck learns patterns
- Decoder upsamples with **skip fusion** → crisp boundaries
- 7-10% higher IoU vs FCN for precision tasks

::right::

```
Input [24, 512×512]
    ↓ Encoder 1 → [64, 256×256]  ↘
    ↓ Encoder 2 → [128, 128×128]  ↘ (skip saves)
    ↓ Encoder 3 → [256, 64×64]    ↘
    ↓ Encoder 4 → [512, 32×32]    ↘
    ↓ Bottleneck → [1024, 32×32]
    ↑ Decoder 4 ← [512, 64×64] ← skip concat
    ↑ Decoder 3 ← [256, 128×128] ← skip concat
    ↑ Decoder 2 ← [128, 256×256] ← skip concat
    ↑ Decoder 1 ← [64, 512×512] ← skip concat
Output [2, 512×512]
```

<style>
  .slidev-code { font-size: 0.85em; line-height: 1.4; }
  h3 { margin-top: 0.5em; }
</style>

---
layout: image-right
image: /Users/scerruti/RETINNA/docs/training_runs/baseline_24ch_epoch20/loss_curves.png
---

# Phase I Results & The Discovery

**What we got:**
- Val IoU: **0.52** (decent)
- Val Acc: **0.86** (respectable)
- Loss: **BCE+Dice** (0.5/0.5 weight)
- LR: **0.0005**
- Training: **20 epochs, ~20 min**

**The problem:**
Tried tuning for class imbalance with `pos_weight=1.5`—performance got **WORSE**, not better.

**Red flag:** Something fundamental was broken, not just hyperparameters.

<style>
  h1 { font-size: 2.2em; margin-bottom: 0.5em; }
  p { font-size: 1.05em; line-height: 1.7; }
  strong { color: #e63946; }
</style>

---

# The Insight: Why It Failed

```
┌─────────────────────────────┐
│   External Labels (CalFire) │  Human-drawn boundaries
│   Masks created separately  │  Different people, processes
└─────────────────────────────┘
               ?
               (unknown time delta)
               ?
┌─────────────────────────────┐
│  Image Information (S2)     │  Spectral data from satellite
│  What pixels actually show  │  Can't know if prev already burned
└─────────────────────────────┘
```

**The mismatch:** 
- Labels = administrative boundaries (Cal Fire fire perimeters)
- Training data = spectral reflectance (Sentinel-2)
- These are NOT the same thing!

**Data quality** is the bottleneck, not the model. Training a spectral model on non-spectral labels is futile.

<style>
  .slidev-code { font-size: 1.1em; line-height: 2; padding: 1em; background: #f5f5f5; margin: 1em 0; }
</style>

---
layout: two-cols
---

# Phase II_01: Analytical Relabeling

::left::

### RdNBR: Relativized dNBR

**Formula:**
```
RdNBR = dNBR / √|NBRpre|
where:
  dNBR = NBRpre - NBRpost
  NBR = (NIR - SWIR) / (NIR + SWIR)
```

**Why RdNBR?**
- Normalizes by pre-fire vegetation
- Better for arid, sparse CA chaparral
- Accounts for landscape heterogeneity
- USGS standard for burn severity

::right::

### USGS MTBS Classes
| Class | RdNBR Range | Count |
|-------|---|---|
| Unburned | < 0.1 | <1% |
| **Low Severity** | 0.1–0.27 | **~90%** |
| Moderate | 0.27–0.44 | <1% |
| High | 0.44–0.66 | <1% |
| **Extreme** | ≥ 0.66 | **~1-2%** |
| Water | MNDWI>0.3 | ~2% |
| Cloud | Blue>0.25 | ~1% |

**Result:** Realistic distribution for CA (sparse veg + extreme burns)

<style>
  h3 { margin-top: 0; }
  .slidev-code { font-size: 0.95em; }
</style>

---

# Phase II_01 Output: 7-Class Distribution

```
Training: 278 samples → 556 images (pre + post)
├─ Pre-fire:  95% Low | 1% Extreme | 2% Water/Cloud
└─ Post-fire: 94% Low | 2% Extreme | 2% Water/Cloud

Validation: 78 samples → 156 images
├─ Pre-fire:  84% Low | 12% Extreme | 3% Water/Cloud  
└─ Post-fire: 81% Low | 14% Extreme | 3% Water/Cloud
```

**QA checks:** ✅
- Post-fire > pre-fire (severity increases)
- Bimodal distribution (realistic for chaparral)
- Cloud/water detection consistent

**Outcome:** 848 analytically-generated training images, labels derived from spectral data, not administrative boundaries.

<style>
  .slidev-code { font-size: 1em; line-height: 1.8; }
  h1 { font-size: 2em; margin-bottom: 0.8em; }
</style>

---
layout: two-cols
---

# Phase II_02: Architecture Redesign

::left::

### Phase I (24 channels)
```
2 timesteps × 12 bands
[Pre_R, Pre_G, Pre_B, ..., Pre_SWIR]
[Post_R, Post_G, Post_B, ..., Post_SWIR]

Concatenated → hardcoded temporal structure
```

### Phase II_02 (8 channels)
```
4 NAIP-compatible bands × 2 timesteps
[Pre_R, Pre_G, Pre_B, Pre_NIR]
[Post_R, Post_G, Post_B, Post_NIR]

Separate → model learns flexible change patterns
```

::right::

### Why 8-channel?
1. **NAIP transfer ready** - NAIP has RGBN, no SWIR
2. **Flexible learning** - Not hardcoded difference
3. **Z-score normalization** - Removes sensor bias
4. **Augmentation-ready** - Geometric transforms work better
5. **Domain shift resilient** - Learns general burn patterns

**Result:** Better positioned for real-world transfer to NAIP imagery

<style>
  .slidev-code { font-size: 0.9em; line-height: 1.5; }
  h3 { margin-top: 0; margin-bottom: 0.3em; }
</style>

---

# Phase II_02 Training Strategy

| Dimension | Approach | Why |
|---|---|---|
| **Loss** | Weighted Cross-Entropy | Per-class weights from distribution (inverse frequency) |
| **Normalization** | Z-score (per-channel) | Removes sensor/seasonal absolute value bias |
| **Augmentation** | Flip, rotate, zoom/crop | Prepares for NAIP domain shift |
| **Scheduler** | ReduceLROnPlateau | factor=0.5, patience=3 → **6.2% val loss improvement** ✓ |
| **Learning Rate** | 1e-3 | Standard for U-Net on this data scale |
| **Batch Size** | 4 | GPU memory constraint, stable gradient estimates |

**Model ready for:** 20+ epochs, then NAIP transfer phase

<style>
  table { font-size: 0.95em; }
  strong { color: #2ecc71; }
</style>

---
layout: image-right
image: /Users/scerruti/RETINNA/docs/phase4_rgb_ir_training/training_history_with_plateau_scheduler.png
---

# Results & PA3 Learning Objectives

**Training curves show:**
- Smooth convergence (scheduler working)
- Validation loss: **-6.2%** improvement
- Validation accuracy: **+2.0%** improvement
- Training stability: Much cleaner trends

**Achieved PA3 objectives:**
- ✅ U-Net architecture (skip connections)
- ✅ Complete training pipeline (data→norm→aug→loss)
- ✅ Hyperparameter tuning (weights, scheduler, LR)
- ✅ Validation strategy (per-class metrics, fold-based)
- ✅ Architectural decisions driven by constraints
- ✅ Data quality analysis (labels vs spectral info)

<style>
  h1 { font-size: 2em; }
  p { font-size: 1em; line-height: 1.6; }
</style>

---

# The Lesson & Next Steps

## **Data is the bottleneck, not the model**

Architecture choices must be driven by real-world constraints:
- **NAIP transfer** → 8-channel input (not 24)
- **Sensor differences** → Z-score normalization
- **Domain shift** → Data augmentation
- **Spectral mismatch** → RdNBR-based labels

### Next: Phase III
- Test on **NAIP imagery** (higher resolution, RGB+IR)
- Evaluate **zero-shot transfer** (no fine-tuning)
- Fine-tune if needed (likely minimal)
- Deploy on real-world burn detection

**Timeline**: Weeks ahead—ready to scale.

<style>
  h2 { color: #e63946; font-size: 2.2em; margin-bottom: 0.3em; }
  h3 { margin-top: 1.2em; font-size: 1.3em; }
</style>

---

# Q&A

---

## Speaker Notes

### Slide 1: Title
**Hook (~15 sec):**
"Burn scar detection sounds straightforward—train a model, predict burned areas. But I spent weeks trying to tune a perfectly good U-Net and kept failing. Turns out the problem wasn't the model. It was the data. This is how I figured that out, and what it took to fix it."

### Slide 1b: Phase I Architecture
**Architecture overview (~60 sec):**
"Started with a standard U-Net on the CaBuAr dataset. CaBuAr gives you 512×512 satellite tiles with pre-fire and post-fire Sentinel-2 imagery—12 spectral bands each. I concatenated the two timesteps to get 24 channels: all the pre-fire data, followed by all the post-fire data.

Why U-Net? The architecture has skip connections. When you downsampled in the encoder, you lose spatial detail—from 512×512 down to 32×32 at the bottleneck. That's 93% of pixels discarded. The skip connections let the decoder access that high-res detail again. So when you're trying to detect a precise burn boundary—especially for acreage calculations, where even 1% error costs thousands—U-Net's skip connections give you the crisp edges you need, not blurry FCN predictions.

31.1 million parameters. Standard stuff. I thought if I just tuned the hyperparameters right, I could get it to work."

### Slide 2: Phase I Results & The Discovery
**Results and the red flag (~60 sec):**
"Trained for 20 epochs on a T4 GPU in Colab, took about 20 minutes. Got a validation IoU of 0.52 and validation accuracy of 0.86. That's... fine. Not amazing, but a reasonable baseline. Loss function was BCE plus Dice, split 50-50, which is standard for imbalanced classes.

So I tried the obvious next step: tweak the class weights. Add `pos_weight=1.5` to weight burned pixels more heavily. Seems reasonable, right? More burned pixels in the loss means the model should focus harder on learning burn characteristics.

Performance got worse. Not slightly worse. Significantly worse.

That's the red flag. If tuning in the sensible direction makes things worse, it's not a hyperparameter problem. Something more fundamental is broken."

### Slide 3: The Insight
**The data problem (~90 sec):**
"I investigated. What are these labels actually measuring?

CaBuAr comes with binary masks from CalFire fire perimeters. Those are administrative boundaries—legal documents defining where a fire burned for insurance and recovery purposes. Completely legitimate, but they're drawn by people, in offices, based on incident reports and survey data. Not derived from the satellite data itself.

Meanwhile, I'm feeding the model spectral data—what the satellite actually sees. Sentinel-2 measures reflectance in 12 different wavelengths.

Here's the problem: the labels and the image data are measuring different things. The mask says 'this is the fire boundary' based on administrative records. The satellite says 'this is where I see vegetation loss' based on spectral indices. These don't perfectly align. There are unburned islands within fire perimeters. There are low-severity patches outside perimeters.

Plus: I have a pre-fire and post-fire image, but I don't know the time delta between images. I don't know if that pre-fire image already captured a previous burn. The masks are binary—burned or not—but spectral damage is continuous.

Training a spectral model on non-spectral labels is fundamentally broken. You're asking the model to learn a mapping from image features to a target that wasn't derived from those features. It's like asking someone to predict the weather from stock prices—they're unrelated.

**This was the turning point.** Data quality is the bottleneck, not architecture. So instead of tuning the model, I fixed the data."

### Slide 4: Phase II_01 Solution
**Spectral relabeling (~75 sec):**
"Solution: use spectral indices to create better labels.

Specifically: Relativized Normalized Burn Ratio, or RdNBR. Formula's in the slide. RdNBR takes the change in vegetation (dNBR) and normalizes it by the pre-fire vegetation amount. Why? Because in sparse landscapes like California chaparral, a small change in low-vegetation pixels doesn't mean the same thing as a small change in forests.

Miller and Thode (2007) showed RdNBR works better for heterogeneous landscapes. That's exactly what California is—sparse shrubland with some patches of dense vegetation.

USGS has established thresholds for RdNBR. Anything above 0.66 is 'extreme severity.' 0.44–0.66 is 'high.' So I applied those thresholds to every pixel in every pre-fire and post-fire image.

Algorithm: for each of 424 pre/post sample pairs, compute RdNBR, classify every pixel into one of 7 classes, then work backwards. Use that RdNBR distribution to calibrate what single-image NBR thresholds should be, then apply those to both pre and post images separately. Get 7-class labels for every image.

**Result:** 424 samples become 848 training images (pre and post), all labeled according to spectral data."

### Slide 5: Phase II_01 Output
**Distribution and validation (~50 sec):**
"Class distribution is bimodal: 94-95% Low Severity, 1-2% Extreme, rest is water/cloud.

Why is that good? Because it's realistic for California. Chaparral is sparse. When a fire hits it, you either get light damage (vegetation recovers) or complete consumption (nothing left to recover). You don't get a lot of moderate damage—the vegetation is already sparse. Administrative labels often have the opposite distribution because they include administrative regions that might not be burned at all.

Quality checks: post-fire images show more severe damage than pre-fire images. ✓ Cloud detection is consistent. ✓ Water detection is consistent. ✓

Now I had labels that match the spectral data. Time to retrain."

### Slide 6: Phase II_02 Redesign
**Architecture redesign (~70 sec):**
"Phase I used all 24 channels: pre-fire R, G, B, NIR, SWIR, water, etc. times 2 timesteps. But I knew Phase III would involve NAIP imagery. NAIP has only 4 bands: RGB and near-infrared. No SWIR.

So instead of using all 12 Sentinel-2 bands, I dropped down to 4: Red, Green, Blue, NIR. Call them RGBN. Pre-fire RGBN (4 channels) plus post-fire RGBN (4 channels) = 8 channels total.

Why does this matter? 

First: it sets us up for transfer learning to NAIP. The model doesn't expect SWIR, so it won't break when we feed it NAIP data.

Second: instead of hardcoding a difference (post – pre), I give the model both images separately. It learns to extract change patterns flexibly. Maybe it learns to look at NIR drop. Maybe it learns RGB color shifts. Maybe it learns combinations. The model discovers what matters, rather than me forcing a subtraction operation.

Third: with separate pre and post, I can apply z-score normalization per channel. Removes sensor bias, seasonal variation, absolute reflectance differences. The model sees normalized change, not raw reflectance.

Fourth: I can augment the data—flip, rotate, crop—which improves generalization."

### Slide 7: Training Strategy
**Implementation details (~60 sec):**
"Weighted cross-entropy loss. Weights are inverse frequency: if Low Severity is 94% of the data, it gets weight 0.01. Extreme is 1%, it gets weight ~100. Balances the training signal.

Z-score normalization: for every channel, subtract the training-set mean, divide by training-set std. Computed once on training data at the start. Applied consistently to train, validation, test.

Data augmentation on training set: random flips (horizontal and vertical), 90-degree rotations, and zoom/crop from 384×384 back to 512×512. Helps the model generalize to different orientations and zoom levels.

Learning rate scheduler: ReduceLROnPlateau. Start at 1e-3. If validation loss doesn't improve for 3 epochs, multiply the learning rate by 0.5. Reduces overfitting, smooths the training curve.

Why a scheduler? Because simple fixed learning rates cause noise. You can get lucky with some batches and unlucky with others. Adaptive learning rate reduces that variance. Validation results show -6.2% validation loss, +2.0% validation accuracy. Worth it.

Batch size 4: limited by GPU memory, but stable enough for reasonable gradient estimates."

### Slide 8: Results & PA3 Objectives
**Training and learning goals (~75 sec):**
"Training history shows what a scheduler does. The loss curve is smooth, not chaotic. Convergence is clear. Validation accuracy is more stable.

We hit all the PA3 learning objectives:

One: understand U-Net architecture. ✓ We discussed skip connections, encoder-decoder symmetry, why it beats FCN for boundary-precise tasks.

Two: complete training pipeline. ✓ Data loading, normalization, augmentation, forward pass, loss computation, backward pass, checkpoint saving—all in place.

Three: hyperparameter tuning. ✓ We experimented with loss weights, learning rates, schedulers. Learned what works and why.

Four: validation strategy. ✓ Fold-based cross-validation, per-class metrics, no data leakage between train/val/test.

Five: architectural decisions driven by constraints. ✓ We didn't pick 8-channel because it sounded good. We picked it because Phase III is NAIP transfer. Conscious design decision, not trend-following.

Six: analyze data critically. ✓ We identified the label-data mismatch, pivoted from administrative to spectral labels, and validated the new distribution.

The training curves show we got here. The model converges, validation loss is lower than the baseline, and the curves are readable—not overfitting, not diverging."

### Slide 9: The Lesson
**Synthesis (~90 sec):**
"**Data is the bottleneck, not the model.**

I could have spent weeks tuning that Phase I model—trying focal loss, changing learning rates, adding dropout, ensemble methods. I would have squeezed maybe 2-3% performance improvement, if I was lucky. But the fundamental problem—labels don't match spectral data—would still exist.

Instead, I invested in data quality. RdNBR-based labels. Per-sample calibration. Validation that post is more severe than pre. Dropped down to NAIP-compatible bands. Added normalization, augmentation, a scheduler. All of that was *enabled* by fixing the data first.

Architectural choices here weren't arbitrary. RGBN only because NAIP has RGBN. Z-score normalization because Sentinel-2 and NAIP have different absolute reflectance. Separate pre/post images because I want flexible change learning, not hardcoded subtraction. Data augmentation because I want the model to generalize to different views, not overfit to Sentinel-2 biases.

The real lesson: understand your problem, understand your data, and let that drive your architecture choices. A mediocre model on good data beats a sophisticated model on bad data.

Next phase: test on NAIP. We've built something that should transfer. That's the real validation."

### Slide 10: Q&A
**Closing (~30 sec):**
"Questions?"

---

## Technical Notes for Presenter

- **Timing**: Slide 1 (~10 sec), 1b (~60 sec), 2 (~60 sec), 3 (~90 sec), 4 (~75 sec), 5 (~50 sec), 6 (~70 sec), 7 (~60 sec), 8 (~75 sec), 9 (~90 sec), Q&A (~30 sec) = ~670 seconds = ~11 minutes. Leave room for natural pacing and audience reactions.

- **Visual strategy**: Heavy on visuals, minimal text on slides (auctioneer style). Speaker notes contain full narrative. Let the images do the talking—the training curves are real data, the architecture diagrams are clean and understandable.

- **Key transition**: Slide 3 is the pivot. Before: "model didn't work, must be hyperparameters." After: "labels are wrong, need spectral data." Make that transition clear.

- **Impact moments**:
  - Slide 3: The insight that labels and data don't match
  - Slide 5: Class distribution is realistic (validation)
  - Slide 8: Results matching all PA3 objectives
  - Slide 9: "Data is the bottleneck"

- **Delivery tone**: Fast, confident, data-driven. Not apologetic about the pivot (it's a strength, not a weakness—shows problem-solving). Celebrate the validation results.

