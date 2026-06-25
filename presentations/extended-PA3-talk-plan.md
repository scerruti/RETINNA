# Extended PA3 Presentation Plan (25-30 minutes)

## Target
- **Duration:** 25-30 minutes
- **Slides:** ~10-12 slides
- **Words per slide:** 50-80 words (more depth than 10-min version)
- **Design principle:** One major concept per slide, deeper technical detail
- **Academic focus:** Learning objectives with detailed technical justification

## Slide Breakdown

### Slide 1: Title & Context
**Title:** From Classroom Assignment to Climate Tech: PA3 to RETINNA

**Content:**
- PA3: UC San Diego CSE 251B assignment (Winter 2021)
- Task: Semantic segmentation on Indian Driving Dataset (27 street-scene classes)
- RETINNA: Research Experience for Teachers in AI
- Your project: Wildfire burn scar detection using Sentinel-2 satellite imagery
- Connection: Same architectural principles, different domain and scale

**Elements:** Title slide, context hierarchy, motivation statement
**Word count:** ~55 words

---

### Slide 2: The Problem Domain
**Title:** Why Wildfire Detection Matters

**Content:**
- Scale: Sentinel-2 captures 10,980 × 10,980 km swaths every 5 days
- Frequency: Rapid detection for emergency response
- Challenge: Distinguishing burned areas from natural shadows, clouds, water
- Class imbalance: Most pixels are unburned (~90%)
- Opportunity: Automated monitoring reduces manual analysis burden

**Elements:** Earth observation imagery example, problem statement
**Word count:** ~50 words

---

### Slide 3: Dataset Overview
**Title:** Sentinel-2 Multispectral Data & Preprocessing

**Content:**
- Bands: 11 (not just RGB) + Temporal dimension
  - Blue, Green, Red (visible light)
  - Near-infrared (vegetation detection)
  - Short-wave infrared (burn severity)
- Normalization: Per-band statistics from CaBuAr training data
- Temporal stacking: Multiple timepoints (T=8 observations per location)
- Output: Each pixel classified as burned (1) or unburned (0)

**Elements:** Band diagram, example multispectral image, data shape diagram
**Word count:** ~60 words

---

### Slide 4: One-Hot Encoding & Data Pipeline
**Title:** From Pixels to Tensors: PA3's Data Transformation

**Content:**
- PA3 approach: Single-channel mask (pixel values 0-26)
- Conversion: Create 27-channel one-hot tensor
- RETINNA approach: 2-channel output (unburned, burned)
- Key insight: One-hot separates class handling from spatial relationships
- Pipeline: Load → Normalize → One-hot → Batch → GPU

**Elements:** Before/after tensor shape visualization, pseudo-code snippet
**Word count:** ~55 words

---

### Slide 5: FCN Architecture (PA3 Reference)
**Title:** Fully Convolutional Network: Encoder-Decoder Design

**Content:**
- Encoder path (downsampling):
  - 5 conv blocks, each with stride=2
  - Channels: 3 → 32 → 64 → 128 → 256 → 512
  - Total downsample factor: 32× (2^5)
- Decoder path (upsampling):
  - 5 deconv (transposed conv) blocks, stride=2
  - Reverses encoder: 512 → 256 → 128 → 64 → 32 → n_classes
- Batch norm after every conv/deconv; ReLU activation between layers

**Elements:** Detailed architecture diagram, layer specifications
**Word count:** ~70 words

---

### Slide 6: U-Net Architecture (RETINNA Implementation)
**Title:** Skip Connections: U-Net Improves FCN

**Content:**
- Motivation: FCN loses fine spatial details in encoder bottleneck
- Solution: Copy encoder feature maps to decoder at matching scales
- Architecture:
  - Input: 11 channels (Sentinel-2 multispectral)
  - Encoder: Same 5-block structure as FCN
  - Skip connections: Connect encoder layer k to decoder layer (5-k)
  - Output: 2 classes (unburned, burned)
- Benefit: Better boundary preservation, higher small-object detection

**Elements:** U-Net diagram with skip connections highlighted, comparison metrics
**Word count:** ~70 words

---

### Slide 7: Training Setup & Loss Function
**Title:** BCE + Dice Loss: Addressing Severe Class Imbalance

**Content:**
- PA3 context: 27 classes, relatively balanced distribution
- RETINNA challenge: 90/10 split (severe imbalance)
- Solution: Hybrid loss function
  - BCE (Binary Cross Entropy): Per-pixel classification loss
  - Dice loss: Region-level overlap metric
  - pos_weight: Parameter to emphasize false negatives
- Rationale: Dice loss prevents model from predicting "unburned" everywhere

**Elements:** Loss formula visualization, comparison chart (CE vs BCE+Dice)
**Word count:** ~65 words

---

### Slide 8: Training Results & Initial Challenge
**Title:** Why False Negatives Emerged (Validation Deep Dive)

**Content:**
- Training observation: Model achieved reasonable IoU on well-lit burns
- Failure case: Lightweight burns on slopes remained undetected
- Root cause: Model biases toward "unburned" prediction (class imbalance)
- Evidence: High precision, low recall (missing subtle signals)
- Lesson: IoU alone masks class-specific performance issues

**Elements:** Loss curve plot, IoU trend, confusion matrix
**Word count:** ~60 words

---

### Slide 9: Hyperparameter Tuning Strategy
**Title:** Iterative Optimization: Batch Size, Learning Rate, Loss Weights

**Content:**
- Batch size exploration: 8, 16, 32 (trade-off: memory vs. gradient stability)
- Learning rate ranges: 0.0001 to 0.001 (Adam optimizer)
- Loss weight tuning: BCE=0.5, Dice=0.5 initially, then adjusted
- pos_weight tuning: From 1.0 → 1.5 → 2.0 (penalize false negatives more)
- Outcome: pos_weight=1.5 achieved best false negative reduction

**Elements:** Hyperparameter sweep results, tuning impact table
**Word count:** ~65 words

---

### Slide 10: FCN vs. U-Net Empirical Comparison
**Title:** Architectural Comparison: Quantitative Results

**Content:**
- Metrics compared:
  - IoU (Intersection over Union) on burned class
  - Precision: Avoid false positives
  - Recall: Minimize false negatives (critical for wildfire detection)
  - Inference speed: Wall-clock time per image
  - Parameter count: Memory footprint
- Results: U-Net superior on recall and boundary preservation
- Trade-off: U-Net has ~3× parameters; acceptable for safety-critical application

**Elements:** Metrics table, bar chart comparison, speed benchmark
**Word count:** ~70 words

---

### Slide 11: Validation Visualization & Analysis
**Title:** Where the Model Succeeds & Fails: Pixel-Level Heatmaps

**Content:**
- False negative heatmap: Predicted unburned, actual burned (red pixels)
- Spatial pattern: Lightweight burns on slopes remain undetected
- Burned area detection (correct): Green pixels
- Unburned areas: Gray (not evaluated)
- Metrics table below heatmap: IoU, Precision, Recall, FN_Rate
- Conclusion: Model learns general patterns; subtle signals need refinement

**Elements:** Pixel-level heatmap image, metrics table, interpretation notes
**Word count:** ~70 words

---

### Slide 12: Learning Objectives Achieved
**Title:** Key Takeaways (Technical Competencies & Insights)

**Content:**
1. **Architecture design:** FCN encoder/decoder principle; skip connections in U-Net
2. **Loss functions:** Class imbalance handling through hybrid BCE+Dice
3. **Evaluation metrics:** IoU, precision, recall; why each matters
4. **Hyperparameter tuning:** Systematic exploration and impact measurement
5. **Domain adaptation:** From PA3's street scenes to satellite multispectral data
6. **Real-world constraints:** Speed, memory, and accuracy trade-offs
7. **Academic to applied:** Translating assignment concepts to research

**Elements:** Numbered checklist, learning pyramid or summary graphic
**Word count:** ~75 words

---

## Visualization Requirements

### For Slide 3 (Sentinel-2 Bands)
- 11-band visualization (as small thumbnails or diagram)
- Color mapping: Blue, Green, Red, NIR, SWIR labeled
- Example "True Color" RGB composite

### For Slide 5 (FCN Diagram)
- Detailed layer diagram showing:
  - Input shape (N, 3, H, W)
  - Each encoder block with output shapes
  - Each decoder block with output shapes
  - Bottleneck shape (N, 512, H/32, W/32)
  - Output shape (N, n_classes, H, W)

### For Slide 6 (U-Net Diagram)
- Same as Slide 5, plus skip connections
- Show concatenation at each decoder level
- Indicate feature map channels at each level

### For Slide 7 (Loss Comparison)
- Formula boxes for CE, BCE, Dice
- Simple graph showing IoU over epochs for pos_weight=1.0 vs 1.5

### For Slide 8 (Training Results)
- Loss curve: Training and validation loss over epochs
- IoU curve: Validation IoU over epochs
- Confusion matrix or recall/precision breakdown

### For Slide 9 (Hyperparameter Sweep)
- Table of results: Different batch_size, LR, pos_weight combinations
- Column: Best IoU, Best Recall, Final Model Performance
- Highlight winning configuration

### For Slide 10 (Comparison)
- Metrics table: FCN vs U-Net (IoU, Precision, Recall, Params, Speed)
- Bar chart overlay for visual comparison

### For Slide 11 (False Negative Heatmap)
- Large pixel-level heatmap (one validation sample)
- Legend: Red (FN), Green (TP), Gray (TN), Yellow (FP)
- Below: Metrics table with IoU, Precision, Recall, FN_Rate

## Notes for Generator
- Use consistent color scheme (red for errors, green for correct, gray for negative)
- Include speaker notes with technical derivations
- Emphasize learning pathway: PA3 concepts → RETINNA adaptation
- Focus on "why things work" rather than just "what we did"
- Academic tone throughout (avoid marketing claims)
