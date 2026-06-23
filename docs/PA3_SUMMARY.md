# PA3 Summary & Analysis
**Date Created**: 2026-06-22  
**Context**: University Assignment (CSE 251B - Computational Vision, Winter 2021)  
**Relevance to RETINNA**: Reference implementation for semantic segmentation architecture  
**Location**: [docs/PA3_SUMMARY.md](PA3_SUMMARY.md) (available to Gary)

---

## GitHub Issues Integration

PA3 concepts are linked to your week-long sprint issues:

| Phase | Concept | GitHub Issue | Action |
|-------|---------|--------------|--------|
| **Day 1** | Dataset loading pattern | [#3 Data Loading](https://github.com/scerruti/RETINNA/issues/3) | Study PA3 dataloader.py; adapt to TorchGeo |
| **Day 1** | Data exploration & statistics | [#4 Exploratory Analysis](https://github.com/scerruti/RETINNA/issues/4) | Understand normalization, one-hot encoding from PA3 |
| **Day 2** | Normalization strategy | [#8 Data Normalization](https://github.com/scerruti/RETINNA/issues/8) | Compute Sentinel-2 band stats (NOT ImageNet) |
| **Day 3** | Architecture template | [#10 Model Architecture](https://github.com/scerruti/RETINNA/issues/10) | Adapt PA3's FCN: 3→11 channels, 27→2 classes |
| **Day 3** | Training loop structure | [#11 Training Script](https://github.com/scerruti/RETINNA/issues/11) | Copy PA3 pattern from starter.py; swap loss function |
| **Day 3** | Evaluation metrics | [#12 Baseline Evaluation](https://github.com/scerruti/RETINNA/issues/12) | Implement IoU focusing on class 1 (burned) only |
| **Day 4** | Hyperparameter ranges | [#14 Hyperparameter Tuning](https://github.com/scerruti/RETINNA/issues/14) | Test batch_size 8–32, LR 0.0001–0.001 |
| **Day 4** | Loss function selection | [#15 Loss Optimization](https://github.com/scerruti/RETINNA/issues/15) | Implement BCE + Dice (handle class imbalance) |

---

## Table of Contents (Priority Ordered)

1. **[Executive Overview](#executive-overview)** — What this assignment covers and why it matters
2. **[Content Inventory](#content-inventory)** — Files, purpose, and completion status
3. **[Learning Objectives](#learning-objectives)** — What students should understand
4. **[Key Concepts & Implementation Tasks](#key-concepts--implementation-tasks)** — Technical requirements
5. **[Comparison to RETINNA Project Plan](#comparison-to-retinna-project-plan)** — Alignment & divergence
6. **[Recommended Reading Order](#recommended-reading-order)** — How to approach this material
7. **[Knowledge Gaps & Next Steps](#knowledge-gaps--next-steps)** — What you'll need to figure out

---

## Executive Overview

**What PA3 Is:**  
A **university assignment from UC San Diego (CSE 251B, Winter 2021)** authored by Gary Cottrell, teaching semantic segmentation via hands-on implementation. Students complete a partially-written PyTorch codebase by filling in missing components marked with `__` placeholders.

**Context:**  
PA3 is **reference/educational material** provided within the **RETINNA program** (Research Experience for Teachers in Interdisciplinary Artificial Intelligence). RETINNA is a 6-week summer program where Gary mentors educators in deep learning. You're developing your own semantic segmentation project (wildfire burn scars) while having access to this pedagogically-structured reference assignment.

**Why PA3 Matters:**  
PA3 demonstrates the core semantic segmentation pipeline you're building for your wildfire project:
- Complete FCN architecture (encoder-decoder pattern)
- End-to-end dataset loading and preprocessing
- Training, validation, and test loops
- Evaluation metrics (IoU, pixel accuracy)
- Industry-standard deep learning patterns on GPU

Unlike a lecture or paper, PA3 is **implementable code**—a concrete blueprint you can study and adapt.

**Scope:**  
The assignment teaches semantic segmentation on the Indian Driving Dataset (IDD, 27 street-scene classes). Completing it from scratch would take ~8–12 hours, but studying it takes far less.

---

## Content Inventory

### Critical Files (Must Read)

| File | Purpose | Status | Pages |
|------|---------|--------|-------|
| `basic_fcn.py` | FCN architecture skeleton | **Incomplete** | 40 lines |
| `dataloader.py` | IDD dataset loader | **Mostly Complete** | 84 lines |
| `starter.py` | Training/validation/test script | **Incomplete** | 81 lines |
| `utils.py` | Metric calculation (IoU, pixel accuracy) | **Incomplete** | 15 lines |

### Data Files

| File | Purpose | Size |
|------|---------|------|
| `train.csv` | Training image-label pairs paths | 746 KB |
| `val.csv` | Validation image-label pairs paths | 373 KB |
| `test.csv` | Test image-label pairs paths | 186 KB |

### Reference Materials

| File | Purpose | Size | Access |
|------|---------|------|--------|
| `CSE_251B__PA3_WI21.pdf` | Official assignment specification | ~400 KB | **Cannot read directly** |
| `.gitignore` | Excluded files (images, etc.) | 453 B | Standard |

---

## Learning Objectives

Based on the assignment structure, students should:

### 1. **Understand FCN Architecture**
   - Encoder (downsampling convolutions + pooling)
   - Decoder (upsampling with transposed convolutions)
   - Skip connections between encoder/decoder levels
   - Batch normalization placement and purpose

### 2. **Implement Complete Forward Pass**
   - Sequentially apply 5 convolutional blocks with batch norm
   - Downsample by 2x at each layer (stride=2)
   - Apply ReLU activation between layers
   - Implement decoder with 5 transposed convolution blocks
   - Output classification layer (1×1 convolution to n_class channels)
   - **Return shape**: (N, 27, H, W) = (batch, classes, height, width)

### 3. **Master Dataset Loading & Preprocessing**
   - Parse CSV files containing image/label paths
   - Load RGB images and single-channel segmentation masks
   - Normalize images: scale [0-255] → [0-1], then apply ImageNet stats
   - Create one-hot encoded targets for all 27 classes
   - Handle DataLoader with appropriate batch sizes and worker counts

### 4. **Build Training Pipeline**
   - Choose appropriate loss function (likely cross-entropy or Dice for segmentation)
   - Initialize weights using Xavier uniform initialization
   - Implement optimization loop with gradient updates
   - Track loss per iteration and per epoch
   - Checkpoint model after each epoch

### 5. **Implement Validation & Testing**
   - Calculate **Intersection over Union (IoU)** per class
   - Calculate **pixel-level accuracy**
   - Apply softmax to model outputs before metric computation
   - Monitor validation performance to prevent overfitting

### 6. **Understand Evaluation Metrics**
   - **IoU (Jaccard Index)**: intersection/union per class
   - **Pixel Accuracy**: (correct pixels) / (total pixels)
   - Why IoU matters for imbalanced datasets (17 non-"unlabeled" + 1 "unlabeled" = 27 total classes)

---

## Key Concepts & Implementation Tasks

### Placeholders to Fill (`__` marks)

| Location | Type | Context | Estimate |
|----------|------|---------|----------|
| `basic_fcn.py` lines 8-29 | Architecture | Input/output channel dimensions for conv/deconv/batchnorm/classifier | Critical |
| `basic_fcn.py` lines 32-36 | Forward logic | Encoder path, batchnorm calls, decoder path | Critical |
| `starter.py` line 19-21 | DataLoader | batch_size, num_workers | Important |
| `starter.py` line 29 | Hyperparameter | epochs | Important |
| `starter.py` line 30 | Loss function | Choose from PyTorch loss functions | Critical |
| `starter.py` line 34 | Learning rate | Optimizer hyperparameter | Important |
| `starter.py` lines 48-51 | GPU handling | Move tensors to device | Important |
| `starter.py` lines 69-77 | Validation/Test | Implement loss/metric calculation | Critical |
| `utils.py` lines 4-11 | IoU metric | Compute intersection & union per class | Critical |
| `utils.py` line 15 | Pixel accuracy | Compute per-pixel correctness | Critical |

### Dataset Details

**IDD (Indian Driving Dataset)**:
- **27 semantic classes**: road, sidewalk, person, motorcycle, car, truck, vegetation, sky, building, etc.
- **Format**: RGB images + single-channel segmentation masks (pixel value = class index)
- **Label encoding**: Each pixel value 0–26 represents one of 27 classes
- **One-hot conversion**: Transform single-channel mask into 27-channel tensor (target shape: (27, H, W))

### Architectural Requirements

**FCN Specifics**:
- **Encoder**: 5 conv blocks, each with stride=2 (downsamples 2×)
  - Input → 32ch → 64ch → 128ch → 256ch → 512ch
  - Total downsample: 32× (2^5)
  
- **Decoder**: 5 deconv blocks with stride=2 (upsamples 2×)
  - 512ch → 256ch → 128ch → 64ch → 32ch → n_class(27)ch
  - Total upsample: 32× (back to original resolution)

- **Activation & Normalization**:
  - ReLU between all conv/deconv blocks
  - Batch normalization after every conv/deconv
  - No activation on final classifier output

---

## Comparison to Your Wildfire Segmentation Project

### Similarities (PA3 ↔ Your Wildfire Segmentation Project)

| Aspect | PA3 (IDD) | Your Project (Wildfire) | Alignment |
|--------|-----------|------------------------|-----------|
| **Task** | Pixel-level semantic segmentation | Pixel-level semantic segmentation | ✅ Identical |
| **Architecture** | Fully Convolutional Network (FCN) | Custom PyTorch U-Net | ⚠️ Similar (U-Net is FCN variant with skip connections) |
| **Input Shape** | RGB 3-channel images | Multi-spectral Sentinel-2 (11 bands) | ⚠️ Channel count differs |
| **Output** | 27-class probability maps | 2-class binary mask (burned/unburned) | ⚠️ Classes differ |
| **Loss Function** | To be chosen (likely CE) | Hybrid BCE + Dice | ✅ Comparable approach |
| **Metrics** | IoU + pixel accuracy per class | IoU specifically for "burned" class | ✅ IoU is primary |
| **Evaluation** | Per-class metrics | Class-specific focus | ✅ Same paradigm |
| **Training Pipeline** | Epochs, batches, gradient descent | Same framework | ✅ Identical |

### Key Differences

| Aspect | PA3 Context | Your Project Context | Impact |
|--------|-------------|----------------------|--------|
| **Dataset Source** | Pre-prepared CSV + local disk | CaBuAr via Hugging Face + TorchGeo | You use modern data library |
| **Input Channels** | 3 (RGB) | 11 (multi-spectral Sentinel-2) | Requires custom input conv layer |
| **Number of Classes** | 27 (street scene semantic labels) | 2 (burned/unburned binary) | Simpler but requires different loss |
| **Class Imbalance** | Likely moderate (varied distribution) | Severe (~90% unburned, 10% burned) | Needs careful loss weighting |
| **Scale & Context** | Urban street scenes | Continental satellite imagery | Different normalization needed |
| **GPU Requirements** | Single GPU sufficient | GPU recommended (Colab) | Training scales with data |

### How PA3 Informs Your Project: Code-Level Mapping

#### **1. Architectural Template (PA3 → Your Wildfire U-Net)**
```
PA3 (FCN):                    RETINNA (U-Net):
Input: 3 channels            Input: 11 channels (Sentinel-2 multispectral)
Encoder: 3→32→64→128→256→512 Encoder: 11→32→64→128→256→512 (same after input)
Decoder: 512→256→128→64→32→27 Decoder: 512→256→128→64→32→2 (27 classes → 2 classes)
Output: 27-class logits      Output: 2-class probabilities (burned/unburned)

**Action**: Copy PA3's encoder/decoder structure; change input Conv2d(3,32,...) → Conv2d(11,32,...) and output Conv2d(32,2,...)
```

#### **2. Dataset Loading (PA3 → Your Project)**
```python
# PA3: CSV-based loader
csv_file → pd.read_csv() → [image_path, label_path] → load from disk

# Your Project: TorchGeo-based loader  
CaBuAr dataset → TorchGeo → automatic download/cache → load from memory/disk

**Action**: Rather than replicating PA3's CSV pattern, use TorchGeo's built-in CaBuAr dataset. 
Study PA3's __getitem__ normalization (lines 72–76) and apply equivalent transforms to TorchGeo data.
```

#### **3. Training Loop (PA3 → Your Project)**
```python
# PA3 pattern (starter.py lines 41–65): REPLICATE EXACTLY
for epoch in range(epochs):
    for iter, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    val(epoch)  # Validation after each epoch

# RETINNA: Use identical loop; only change:
# - Loss function: BCE+Dice (not just CrossEntropy)
# - Batch size: likely smaller (GPU memory for 11-ch input)
# - Learning rate: tune empirically (PA3 may differ from optimal for 2-class problem)
```

#### **4. Evaluation Metrics (PA3 → RETINNA)**
```python
# PA3 (utils.py): Compute IoU for all 27 classes
def iou(pred, target):
    for cls in range(27):  # All 27 classes
        intersection = (pred == cls) & (target == cls)
        union = (pred == cls) | (target == cls)
        ious.append(intersection/union)
    return ious

# RETINNA: Compute IoU for class=1 (burned) ONLY
def iou_burned(pred, target):
    cls = 1  # Burned class
    intersection = (pred == cls) & (target == cls)
    union = (pred == cls) | (target == cls)
    return intersection / union  # Single scalar, not list

**Action**: Copy PA3's per-class loop structure; iterate over classes [0,1] or just focus on cls=1.
```

#### **5. Loss Function (PA3 → RETINNA)**
```python
# PA3: Open choice (likely CrossEntropyLoss for 27 classes)
criterion = nn.CrossEntropyLoss()

# Your Project: Hybrid loss for class imbalance
criterion = BCE_loss + alpha*Dice_loss  # where alpha~0.5–1.0

**Why**: PA3's 27 classes are relatively balanced. Your 90/10 split (unburned vs burned) requires loss that doesn't collapse to predicting "unburned" everywhere.

**Action**: Use PyTorch's BCEWithLogitsLoss() or BCELoss() + custom Dice. Your README specifies "hybrid loss combining BCE + Dice".
```

#### **6. Hyperparameter Ranges (PA3 → Your Project)**

| Hyperparameter | PA3 Typical Range | RETINNA Expected | Reason |
|---|---|---|---|
| **Batch Size** | 16–64 | 8–32 | 11-channel input uses more GPU memory |
| **Learning Rate** | 0.0001–0.001 | 0.0001–0.001 | Same range likely works |
| **Epochs** | 50–200 | 30–100 | Fewer classes (2 vs 27) → faster convergence |
| **Optimizer** | Adam (shown) | Adam | Industry standard; no change needed |
| **Weight Init** | Xavier Uniform (shown) | Xavier Uniform | Use PA3's pattern verbatim |

#### **7. Normalization (PA3 → Your Project)**
```python
# PA3: ImageNet statistics (standard for RGB)
transforms.Normalize((0.485, 0.456, 0.406),  # RGB means
                     (0.229, 0.224, 0.225))  # RGB stds

# Your Project: Sentinel-2 multispectral statistics
# Sentinel-2 has 11 bands, NOT RGB-only
# You'll need to compute band-wise mean/std from CaBuAr training data
# OR use simple min-max scaling: (x - x.min()) / (x.max() - x.min())

**Action**: Compute Sentinel-2 normalization stats from CaBuAr dataset.
Don't copy ImageNet stats—they won't apply to multispectral imagery.
```

### How PA3 Informs Your Project: Conceptual Mapping

| PA3 Concept | How It Applies to Your Work | Adaptation Required |
|-----------|------------------------|---|
| **FCN architecture design** | Use encoder/decoder structure as template | ✏️ Change input channels 3→11, output classes 27→2 |
| **Dataset loading pattern** | Understand iteration/batching paradigm | ✏️ Use TorchGeo instead of CSV |
| **Training loop structure** | Copy loop, backward pass, optimizer.step() | ✅ No change; use pattern verbatim |
| **Metric computation** | IoU/accuracy calculation transferable | ✏️ Focus on "burned" (class 1) IoU only |
| **Batch normalization** | Include after conv/deconv layers | ✅ Include in U-Net; same placement |
| **Weight initialization** | Xavier uniform for conv/deconv | ✅ Use PA3's approach; copy directly |
| **Validation strategy** | Track loss + metrics per epoch, use early stopping | ✅ Replicate PA3's approach for your project |
| **Loss function choice** | CrossEntropy works for PA3's 27 classes | ✏️ Use Dice or BCE+Dice for your 2-class imbalance |
| **GPU handling** | torch.cuda.is_available() + model.cuda() | ✅ Use PA3's pattern; same for Colab |

### RETINNA Advantages Over PA3

| Feature | Status in PA3 | Status in RETINNA | Benefit |
|---------|---------------|-------------------|---------|
| **Modern data library (TorchGeo)** | Manual CSV-based loading | Automatic via TorchGeo | Reduced data engineering |
| **U-Net skip connections** | Implicit in FCN design | Explicit in RETINNA architecture | Better boundary preservation |
| **Real-world domain** | Synthetic urban dataset | Actual satellite data | Practical applicability |
| **Hybrid loss function** | To be chosen by student | Explicit BCE + Dice | Better handles imbalance |
| **Cloud compute** | Local GPU assumed | Colab GPU available | Scalable training |

---

## Recommended Reading Order

### **Phase 1: Understand the Problem (15-20 min)**
1. Read this document's sections 1–2 (Executive Overview + Content Inventory)
2. **Skim** `CSE_251B__PA3_WI21.pdf` (assignment spec) — focus on:
   - Task description
   - What constitutes a "complete" submission
   - Any specific hyperparameter recommendations
   - Grading rubric (reveals what they're testing)

### **Phase 2: Learn the Architecture (30-40 min)** → [GitHub Issue #10](https://github.com/scerruti/RETINNA/issues/10)
3. Study `basic_fcn.py`:
   - Trace the layer sizes: 3ch → 32 → 64 → 128 → 256 → 512 → (decoder reverses) → 27ch
   - Understand stride=2 downsampling (32× total)
   - Note batch norm placement
   - Identify where forward pass is incomplete
4. Read the "Key Concepts & Implementation Tasks" section above (highlights what's missing)

### **Phase 3: Understand the Data (20-30 min)** → [GitHub Issues #3, #4, #8](https://github.com/scerruti/RETINNA/issues/3)
5. Study `dataloader.py`:
   - Note the 27 label classes (lines 18–47)
   - Understand ImageNet normalization (line 59-60) — **KEY**: This won't apply to your Sentinel-2 data
   - Follow `__getitem__` logic: load RGB image, label mask, create one-hot tensor
   - CSV format: [image_path, label_path]

### **Phase 4: Trace Training Workflow (20-30 min)** → [GitHub Issues #11, #12](https://github.com/scerruti/RETINNA/issues/11)
6. Study `starter.py` (sketch, not complete):
   - Lines 14–21: DataLoader initialization (has `__` blanks)
   - Lines 24–34: Model init, criterion, optimizer (has `__` blanks)
   - Lines 41–65: `train()` function (sketch with `__` blanks)
   - Lines 69–77: `val()` and `test()` stubs (completely empty)

### **Phase 5: Understand Evaluation (15-20 min)** → [GitHub Issue #12](https://github.com/scerruti/RETINNA/issues/12)
7. Study `utils.py`:
   - `iou()` function: understand intersection/union per class
   - `pixel_acc()` function: understand per-pixel correctness
   - These are the metrics you'll report in validation/testing

### **Phase 6: Hyperparameters & Loss (10-15 min)** → [GitHub Issues #14, #15](https://github.com/scerruti/RETINNA/issues/14)
8. Study hyperparameter ranges and loss function choices
   - Note what transfers directly (architecture, loss paradigm, metrics)
   - Identify what differs (channels, classes, dataset, loss weighting)
   - Understand class imbalance handling: PA3's 27 classes vs your 2-class problem

**Total Time**: ~2 hours of focused reading

---

## Knowledge Gaps & Next Steps

### Questions the PDF Assignment Likely Answers

You need to **read `CSE_251B__PA3_WI21.pdf`** to fill these gaps. The PDF probably contains:

| Question | Likely Answers in PDF | Impact if Missed |
|---|---|---|
| **Loss function choice** | "Use CrossEntropyLoss" or "Implement custom Dice loss" | Wrong choice = model trains slowly or doesn't converge |
| **Batch size guidance** | "Use 16 or 32" or "Adjust based on GPU memory" | Too large → OOM; too small → noisy gradients |
| **Learning rate** | "Start with 0.0001" or "Use learning rate schedule" | Too high → unstable training; too low → slow convergence |
| **Epoch count** | "Train for 50–100 epochs" or "Use early stopping" | Too few → underfitting; too many → overfitting |
| **Number of workers** | "Use 4 workers" or "Use num_workers=(CPU cores - 1)" | Too many → deadlocks; too few → slow data loading |
| **Evaluation target** | "Target IoU > 0.5" or "Any IoU > 0.3 acceptable" | Tells you if your model is working |
| **Specific `__` values** | Exact channels/dimensions for conv layers | Without these, forward() won't run |
| **Submission format** | "Save as .pth checkpoint" and "Include inference script" | Grading depends on format compliance |
| **GPU requirements** | "Must run on GPU" or "CPU acceptable but slow" | Informs whether to use local GPU or Colab |
| **Dataset download** | "Data provided in zip" or "Download from link" | May need to adjust dataloader paths |

**Action**: Print the PDF and annotate it. Gary likely provides concrete values for rows 2–7 above.

### Implementation Challenges You'll Likely Face

1. **Filling the forward pass** (basic_fcn.py, lines 32-36)
   - You'll need to trace through layer dimensions carefully
   - Encoder: apply conv → batchnorm → relu in sequence, 5 times
   - Decoder: apply deconv → batchnorm → relu, 5 times
   - Final classifier: single 1×1 conv to output n_class channels

2. **Choosing loss function** (starter.py, line 30)
   - Options: `nn.CrossEntropyLoss()`, `nn.BCELoss()`, custom Dice loss
   - IDD has 27 classes → probably `CrossEntropyLoss` (expects logits)
   - RETINNA has 2 classes → BCE or Dice (expects probabilities)

3. **GPU memory management**
   - IDD images are street scenes (typically 512×512 or higher)
   - 32 batch size might hit GPU memory limits on older cards
   - Adjust batch size if you see OOM errors

4. **Validation metric computation**
   - Must apply `softmax` or `argmax` before computing IoU
   - Watch for edge cases (classes with 0 pixels in ground truth)
   - Be careful not to mix batch dimensions in metric calculation

5. **Dataset imbalance**
   - "unlabeled" (class 26) may dominate; loss function should handle this
   - RETINNA has worse imbalance (10% vs 90%) → Dice loss may be critical

### How to Verify Your Implementation (Debugging Checklist)

**Step 1: Model Architecture Verification**
- [ ] Model loads without errors: `model = FCN(n_class=27)`
- [ ] Forward pass completes: `output = model(torch.randn(2, 3, 512, 512))`
- [ ] Output shape is correct: `(2, 27, 512, 512)` for batch=2, classes=27, spatial=512×512
- [ ] Model is on GPU (if applicable): `model.cuda()` doesn't error

**Step 2: Data Loading Verification**
- [ ] IddDataset loads one sample: `dataset[0]` returns (img, target, label) triple
- [ ] Image shape: `(3, H, W)` with values ~[0, 1] (normalized)
- [ ] Target shape: `(27, H, W)` with binary one-hot encoding
- [ ] Label shape: `(H, W)` with integer class indices 0–26
- [ ] DataLoader batching works: `next(iter(train_loader))` returns 4-tuples (batch_img, batch_target, batch_label, ...)

**Step 3: Training Loop Verification**
- [ ] Model runs in train mode: `model.train()`
- [ ] Forward pass: `output = model(inputs)` completes
- [ ] Loss computation: `loss = criterion(output, labels)` doesn't error
- [ ] Backward pass: `loss.backward()` doesn't error
- [ ] **First epoch loss is reasonable**: ~log(27)≈3.3 for random CrossEntropy
- [ ] **Loss decreases** from epoch 1 → epoch 2 (not increasing)
- [ ] Model can be saved: `torch.save(model, 'test.pth')`

**Step 4: Validation Metrics Verification**
- [ ] IoU function computes: `iou_scores = iou(predictions, ground_truth)`
- [ ] Returns list of 27 values: `len(iou_scores) == 27`
- [ ] Values are in [0, 1]: `all(0 <= s <= 1 for s in iou_scores)`
- [ ] Values aren't all NaN (means no ground truth for a class, OK)
- [ ] Mean IoU is non-zero: `np.nanmean(iou_scores) > 0.01`
- [ ] Pixel accuracy computes: `acc = pixel_acc(predictions, ground_truth)`
- [ ] Accuracy is in [0, 1]: `0 <= acc <= 1`
- [ ] **Baseline accuracy is reasonable**: ~(dominant class frequency). Road is common, so expect ~30–50% random baseline

**Step 5: Training Progress Verification**
```
Epoch 1: loss 3.2, val_iou 0.02, val_acc 0.35
Epoch 2: loss 2.8, val_iou 0.04, val_acc 0.38  ← Loss decreasing, metrics increasing (good!)
Epoch 3: loss 2.5, val_iou 0.06, val_acc 0.40
...
```

### Common Bugs & How to Fix Them

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| **Model shape error** | `__` placeholder still in architecture | Search for `__` in `basic_fcn.py`; fill in channel dimensions |
| **Loss is NaN** | Loss function expects different input shape | Verify `criterion(output, labels)` where output=(N,27,H,W) and labels=(N,27,H,W) one-hot |
| **Loss doesn't decrease** | Learning rate too small, or model not learning feature | Increase LR (e.g., 0.0001→0.001); check that gradients are flowing |
| **GPU OOM error** | Batch size too large for GPU memory | Reduce batch_size from 32→16 or 16→8 |
| **IoU is always 0** | Predictions and ground truth don't overlap; or metric implementation wrong | Visualize predictions vs ground truth; check metric logic |
| **Dataloader hangs** | num_workers too high or data path wrong | Set num_workers=0 for debugging; check CSV file paths are absolute or correct relative paths |
| **Model loads but errors on forward pass** | Dimension mismatch in conv layers | Print input shape, then each layer's output shape; trace where mismatch occurs |
| **Validation slower than training** | eval() mode might not be set, or validation dataset is large | Add `model.eval()` before validation; check val_dataset size |

---

## Summary: What You Should Be Looking For

### ✅ Essential Takeaways from PA3

| Concept | Why It Matters | PA3 Example |
|---------|---|---|
| **FCN encoder/decoder symmetry** | Guarantees spatial reconstruction; no information loss | 3→32→64→128→256→512 downsamples; 512→256→128→64→32→output upsamples |
| **Batch normalization placement** | Stabilizes training; prevents internal covariate shift | After every conv/deconv, before ReLU (PA3 does this correctly) |
| **One-hot encoding** | Converts single-channel mask (pixel values 0–26) into 27-channel tensor | See `dataloader.py` lines 78–82; replicate for RETINNA's 2-class case |
| **Per-class IoU metric** | Handles class imbalance; IoU for class=1 (burned) is RETINNA's primary metric | `utils.py` iou() loops over all classes; RETINNA will focus on class 1 |
| **Training loop with validation** | Monitors generalization; enables early stopping on validation performance | `starter.py` lines 41–65: epoch loop with per-epoch validation call |
| **GPU memory management** | Batch size, image resolution, and channel count interact; OOM errors signal need to reduce batch size | PA3 uses 3-channel images; RETINNA uses 11-channel → expect ~3.7× higher GPU usage per sample |

### 🔗 Direct Transferability to RETINNA

**Copy Directly (No Changes)**:
- Training loop structure (forward → loss → backward → step)
- Validation loop concept (eval mode, loss/metric computation, train mode reset)
- Weight initialization pattern (Xavier uniform)
- GPU handling (torch.cuda.is_available(), model.cuda())
- Learning rate range (0.0001–0.001)
- Optimizer choice (Adam)

**Adapt with Simple Changes**:
- **Input channels**: `Conv2d(3, 32, ...)` → `Conv2d(11, 32, ...)`
- **Output classes**: `classifier = Conv2d(32, 27, ...)` → `classifier = Conv2d(32, 2, ...)`
- **Loss function**: `CrossEntropyLoss()` → `BCEWithLogitsLoss() or custom Dice()`
- **Batch size**: Start at 32; reduce to 16 or 8 if GPU OOM
- **Normalization**: Replace ImageNet stats with Sentinel-2 band statistics

**Replace Entirely**:
- Dataset loading: IDD CSV → TorchGeo CaBuAr
- Normalization values: Compute from Sentinel-2 data, not ImageNet
- IoU metric: Focus on class=1 (burned) only, not all 27 classes

### 📋 Pre-Sprint Checklist (Before Day 1)

- [ ] **Read this document** (you are here)
- [ ] **Skim the PA3 PDF** (CSE_251B__PA3_WI21.pdf) for assignment requirements
- [ ] **Study `dataloader.py`** (understand one-hot encoding, normalization pattern) → [#3](https://github.com/scerruti/RETINNA/issues/3), [#4](https://github.com/scerruti/RETINNA/issues/4)
- [ ] **Study `basic_fcn.py`** (trace layer sizes, understand encoder/decoder symmetry) → [#10](https://github.com/scerruti/RETINNA/issues/10)
- [ ] **Identify the gaps** in `starter.py` you'd need to fill if implementing PA3 → [#11](https://github.com/scerruti/RETINNA/issues/11)
- [ ] **Understand `utils.py`** (per-class IoU calculation, pixel accuracy concept) → [#12](https://github.com/scerruti/RETINNA/issues/12)
- [ ] **Review hyperparameter strategy** for your project (batch size, learning rate, loss) → [#14](https://github.com/scerruti/RETINNA/issues/14), [#15](https://github.com/scerruti/RETINNA/issues/15)

### 🚀 How to Use PA3 for Your Project Development

1. **Dataset Phase**: Study PA3's `dataloader.py` pattern (CSV loading, normalization, one-hot encoding). Adapt concepts to TorchGeo's CaBuAr dataset loading.
2. **Architecture Phase**: Use PA3's FCN encoder/decoder as a template for your U-Net. Change input channels (3→11) and output classes (27→2).
3. **Training Phase**: Copy PA3's training loop structure verbatim. Swap loss function to Dice or BCE+Dice for your severe class imbalance.
4. **Evaluation Phase**: Replicate PA3's test function and IoU metric code. Focus your metrics on the "burned" class (class 1) rather than all classes.

---

**Generated by Claude Code**  
**Context**: PA3 is reference material within the RETINNA educational program. This summary helps you understand how PA3's semantic segmentation patterns apply to your wildfire burn scar project.

**Revision**: Iteration 3 — Corrected to reflect that RETINNA is an educational program, not your project name. PA3 is pedagogical reference material for learning deep learning concepts.
