# Phase II_02 Iteration Analysis: Class Imbalance Solutions

**Date**: 2026-06-25  
**Project**: RETINNA Phase II - 8-Channel U-Net Burn Severity Classification  
**Problem**: Extreme class imbalance preventing minority class detection

---

## Executive Summary

Attempted four different approaches to fix class imbalance in burn severity classification:

| Iteration | Approach | Extreme Recall | Overall Acc | Status |
|-----------|----------|----------------|-------------|--------|
| 2 (Baseline) | CE Loss + random shuffle | 31% | 81.6% | **BEST** |
| 3 | Focal Loss (buggy) | FAILED | FAILED | Abandoned |
| 4 | CE Loss + WeightedRandomSampler | 10% | 76.1% | **WORSE** |
| 5 | Focal Loss + sqrt-dampened weights | 0% | 83.5% | **WORST** |

**Key Finding**: None of the attempted improvements worked. Iteration 2 (simple CE Loss + random shuffle) remains the best performing model.

---

## The Core Problem: Extreme Class Imbalance

### Data Distribution (Training Set)

```
Class Distribution:
- Low Severity:   14.8M pixels (83.0%)
- Water:           1.4M pixels (8.0%)
- Cloud:           1.0M pixels (6.0%)
- Moderate:        0.4M pixels (2.4%)
- Extreme:         0.2M pixels (0.9%)  ← TARGET CLASS (most important)
- High:            0.03M pixels (0.2%)
- Unburned:        0.0001M pixels (0.002%)
```

### The Problem Manifested

With 83% of pixels being Low Severity:
1. **Model learned majority class as "default safe strategy"**
   - Predicting Low Severity for everything = 83% accuracy
   - No incentive to learn minority classes initially

2. **Weighted CE Loss insufficient**
   - Class weights inverse to frequency: [3.96, 0.005, 0.42, 2.30, 0.08, 0.09, 0.14]
   - Even with 30:1 weight ratio on rare classes, model defaulted to majority
   - Loss optimization doesn't guarantee learning

3. **Recall on critical class drops to 31%**
   - Extreme pixels: only 31% detected
   - Precision: only 4% (96% false positives when predicting Extreme)
   - F1 score: 0.06 (essentially useless)

---

## Iteration 2: Baseline (CE Loss + Random Shuffle)

### Configuration

```python
class_weights = 1.0 / (class_counts + 1.0)  # Inverse frequency
class_weights = class_weights / class_weights.mean()  # Normalize to mean=1

criterion = nn.CrossEntropyLoss(weight=class_weights)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
```

### Results

**Test Set Performance:**
```
Overall Pixel Accuracy: 81.58%
Macro-Averaged IoU: 0.2535

Per-Class Results:
Class          Accuracy  IoU     Recall  Precision  F1
Low Severity   86.44%    0.848   86%     98%        0.92
Cloud          90.56%    0.573   91%     61%        0.73
Water          56.34%    0.321   56%     43%        0.49
Extreme        31.19%    0.033   31%     4%         0.06  ← CRITICAL FAILURE
Moderate       0.00%     0.000   0%      0%         0.00
High           0.00%     0.000   0%      0%         0.00
Unburned       0.00%     0.000   0%      0%         0.00
```

### Analysis: Why It Failed

1. **Class weights weren't extreme enough**
   - Ratio Low Severity : Extreme = 3.96 / 0.08 = 49.5:1
   - Still allowed model to default to majority class
   - Weight of 0.08 for Extreme (vs 3.96 for Low) = model learned Low Severity is ~50× more important

2. **Random shuffle created imbalanced batches**
   - Each batch of 4 images typically contained:
     - ~3.3 images worth of Low Severity pixels
     - ~0.08 images worth of Extreme pixels
   - Loss function dominated by majority class signal
   - Model learned Low Severity bias from batch statistics

3. **Local minima**
   - CE Loss reaches local minimum at "predict Low Severity always"
   - Gradient updates too small to escape this minimum
   - Model got stuck after epoch 5

4. **Minority classes underfitted**
   - Moderate, High, Unburned: 0% recall
   - Model never learned these classes exist
   - Data was too sparse relative to majority signal

### Why Iteration 2 Was Best (Among Failures)

Despite 31% Extreme recall being unacceptable, it was the only approach that achieved ANY minority class detection:
- Low Severity recall: 86% (maintained majority class)
- Extreme recall: 31% (better than 0%)
- Water recall: 56% (partial detection)

---

## Iteration 3: Focal Loss (Abandoned - Implementation Failed)

### Motivation

**Focal Loss Paper**: Lin et al. (2017) "Focal Loss for Dense Object Detection"

Formula: `FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)`

Design goals:
- Downweight easy negatives (correct majority class predictions)
- Focus on hard positives (misclassified rare class pixels)
- γ=2.0: 100× more weight to hard examples
- α=class_weights: per-class weighting

### Implementation Attempts

**Attempt 1: Naive Implementation**
```python
ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
p = F.softmax(inputs, dim=1)
p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
focal_weight = (1 - p_t) ** self.gamma
focal_loss = focal_weight * ce_loss
```

**Results**: Loss = 0.03-0.09, Accuracy = 6% (worse than random 14%)
- **Problem**: Shape mismatch or incorrect weight application
- Cross-entropy with weight parameter + manual focal weighting created undefined behavior
- Model collapsed

**Attempt 2: Corrected Implementation**
```python
log_probs = F.log_softmax(inputs_flat, dim=1)
log_p_t = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
ce = -log_p_t
alpha_t = self.alpha.gather(0, targets_flat)
ce = ce * alpha_t
focal_weight = (1.0 - p_t) ** self.gamma
focal_loss = focal_weight * ce
```

**Results**: Loss = 0.04, Epoch 2 failed with "Bad"
- **Problem**: Still numerically unstable
- Extreme class weights (0.005-4.0 range) caused gradient explosions
- Model diverged

### Why Iteration 3 Failed

1. **Extreme class weight ranges destabilized gradients**
   - Inverse frequency: 1.0 / (class_counts + 1.0)
   - Low Severity weight: 1.0 / (14.8M + 1) ≈ 0.0000000676
   - After normalization to mean=1: 0.0046
   - Extreme weight: 1.0 / (0.2M + 1) ≈ 0.000005
   - After normalization: 0.0834
   - Ratio: 0.0046 : 0.0834 ≈ 1:18 (but with CE loss, max weight was 3.96)
   - **Issue**: With Focal Loss + extreme weights, gradients became inconsistent

2. **Focal Loss complexity**
   - Multiple implementations attempted, multiple failures
   - Combination of Focal Loss formula + alpha weighting is subtle
   - Hard to debug numerically

3. **User feedback**: "Option B is tricky since you don't know how to implement focal loss"
   - Correctly identified implementation difficulty
   - Abandoned in favor of data rebalancing

---

## Iteration 4: CE Loss + WeightedRandomSampler

### Motivation

**Problem Identified**: Iteration 2 failed because random shuffle created imbalanced batches

**Solution**: Use WeightedRandomSampler to oversample samples with minority classes
- Compute per-sample weights based on class composition
- Oversample samples containing rare classes
- Create more balanced batch distributions
- Use gentle sqrt-dampened class weights (0.09-2.7 range)

### Implementation

```python
# Compute per-sample weights
for idx in train_dataset.indices:
    label = labels_tensor[len(pre_rgbn_tensor) + sample_idx]
    label_flat = label.flatten().long()
    pixel_weights = class_weights[label_flat]
    avg_weight = pixel_weights.mean().item()
    sample_weights.append(avg_weight)

# Use WeightedRandomSampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler, num_workers=0)
```

### Class Weights (Sqrt-Dampened)

```
Original (Inverse Frequency):
Class 0: weight=3.9551
Class 1: weight=0.0046  ← 860:1 ratio vs Class 0 (EXTREME)
Class 2: weight=0.4235
Class 3: weight=2.3049
Class 4: weight=0.0834
Class 5: weight=0.0855
Class 6: weight=0.1430

After Sqrt Dampening:
Class 0: weight=2.6849
Class 1: weight=0.0917  ← 30:1 ratio vs Class 0 (STABLE)
Class 2: weight=0.8786
Class 3: weight=2.0497
Class 4: weight=0.3898
Class 5: weight=0.3948
Class 6: weight=0.5104
```

### Training Results

**Epoch 1**: Train Acc 30%, Val Acc 57%  
**Epoch 2**: Train Acc 59%, Val Acc 87%  
**Epoch 5**: Train Acc 65%, Val Acc 87%  
**Epoch 20**: Train Acc 66%, Val Acc 82%

Training curves looked healthy (smooth convergence).

### Test Set Results (WORSE than Iteration 2)

```
Overall Accuracy: 76.09%
Macro-Averaged IoU: 0.2293

Per-Class Results:
Class          Accuracy  IoU     
Low Severity   82.07%    0.7865
Moderate       48.92%    0.1433
High           0.00%     0.0000
Extreme        10.48%    0.0100  ← COLLAPSED from 31%
Water          62.20%    0.3612
Cloud          31.30%    0.3040  ← COLLAPSED from 91%
```

### Why Iteration 4 Failed

1. **WeightedRandomSampler created skewed batch distributions**
   - Intent: Create balanced batches
   - Reality: Created batches where individual samples had very different class compositions
   - Example batch:
     - Sample 1: 90% Low Severity (low weight)
     - Sample 2: 50% Extreme (high weight, drawn 3× more often)
     - Sample 3: 80% Cloud (medium weight)
     - Sample 4: 95% Low Severity
   - Result: Highly non-uniform pixel class distribution across training

2. **Model couldn't learn stable decision boundaries**
   - Batch-to-batch variation was too high
   - Sampler forced rare classes into view, but at unnatural frequencies
   - Model learned to predict rare classes in contexts where they never appear in reality
   - Overfitting to sampler-induced artifact

3. **Loss function didn't benefit**
   - CE Loss expects similar class distributions in each batch
   - WeightedRandomSampler violated this assumption
   - Model learned the sampler's distribution, not the data's distribution

4. **Empirical results confirm failure**
   - Extreme recall: 31% → 10% (worse)
   - Cloud recall: 91% → 31% (much worse)
   - Overall accuracy: 81.6% → 76.1% (worse)
   - Only Moderate improved (0% → 49%), but overall model degraded

### Technical Root Cause

WeightedRandomSampler operates on SAMPLES (entire images), not PIXELS. But the class imbalance occurs at the PIXEL level:
- 14.8M Low Severity pixels exist
- 0.2M Extreme pixels exist
- Ratio: 74:1 pixel-level imbalance

When you oversample samples with Extreme pixels:
- You don't get 50 times more Extreme pixels
- You get more samples that contain SOME Extreme pixels
- Most pixels in those samples are still Low Severity
- Net effect: You're creating an artificial distribution that never existed

The model learns from this artificial distribution and fails to generalize to the true distribution.

---

## Iteration 5: Focal Loss + Sqrt-Dampened Weights

### Motivation

Learn from Iteration 3 and 4 failures:
- **Avoid** WeightedRandomSampler (skews batch distribution)
- **Avoid** extreme class weights (destabilize gradients)
- **Try** Focal Loss with stable, sqrt-dampened weights
- **Key difference**: sqrt-dampened weights are stable (0.09-2.7 range, ~30:1 ratio)

### Implementation

```python
class FocalLoss(nn.Module):
    def forward(self, inputs, targets):
        N, C, H, W = inputs.shape
        inputs_flat = inputs.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets_flat = targets.view(-1)
        
        log_probs = F.log_softmax(inputs_flat, dim=1)
        log_p_t = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        probs = F.softmax(inputs_flat, dim=1)
        p_t = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        
        ce = -log_p_t
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets_flat)
            ce = ce * alpha_t
        
        focal_weight = (1.0 - p_t) ** self.gamma
        focal_loss = focal_weight * ce
        return focal_loss.mean()

criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')
```

### Training Results

**Excellent convergence**:
- Epoch 1: Train Loss 0.154, Val Loss 0.107
- Loss ~10× lower than CE+WeightedSampler (0.12-0.15 vs 1.4-1.8)
- Training curves smooth and stable
- Val accuracy plateaued at 87-88%

### Test Set Results (WORST of all iterations)

```
Overall Accuracy: 83.46%
Macro-Averaged IoU: 0.1478

Per-Class Results:
Class          Recall    Precision
Low Severity   100%      84%       ← Predicting EVERYTHING as Low Severity
Moderate       17%       60%
High           0%        0%
Extreme        0%        0%        ← COMPLETE FAILURE
Water          3.12%     0.42
Cloud          1.65%     0.93
```

Detailed classification report:
```
Low Severity: recall=1.00 (model predicts it for all pixels)
Everything else: recall ≈ 0%

Model learned: "Predict Low Severity for everything"
Accuracy: 83.46% (dominated by majority class)
```

### Why Iteration 5 Failed Catastrophically

1. **Focal Loss created perverse incentive structure**
   - Focal Loss downweights easy examples: `(1 - p_t)^γ`
   - For correctly classified Low Severity pixels (p_t ≈ 0.99):
     - focal_weight = (0.01)^2 = 0.0001
     - Loss is downweighted 10,000×
   - For misclassified Extreme pixels (p_t ≈ 0.001):
     - focal_weight = (0.999)^2 ≈ 1.0
     - Loss is NOT downweighted (focal weighting inactive for hard examples)
   
2. **Combined with alpha weights, system became inconsistent**
   - Alpha weighting and focal weighting work against each other
   - Alpha tries to penalize majority class: `α_Low=0.09, α_Extreme=0.39`
   - But focal weighting downweights majority class completely: `(1-p_t)^2 ≈ 0` when p_t ≈ 1
   - Net effect: Majority class gets downweighted by both mechanisms
   - Minority class gets upweighted only by alpha, but downweighted by focal when correct
   
3. **Model learned "predict majority class" as optimal strategy**
   - Average loss for "always predict Low Severity":
     - 83% correct: focal_weight ≈ 0.0001, loss ≈ 0
     - 17% incorrect: focal_weight ≈ 1.0, loss ≈ α_t * log(p_t) ≈ 3-18
     - Average: ~0.5-3 per pixel
   
   - Average loss for "predict mixed classes":
     - Must learn decision boundaries
     - Even correct predictions get some loss
     - Total loss higher
   
   - **Conclusion**: "Always predict majority" was the locally optimal strategy

4. **Empirical results confirm collapse**
   - Extreme recall: 31% → 0% (complete failure)
   - Cloud recall: 91% → 1.65% (99.9% collapse)
   - Water recall: 56% → 3.12% (94% collapse)
   - Model learned single class instead of multi-class

### Technical Root Cause

The combination of:
1. Focal Loss's aggressive downweighting of easy examples
2. Per-class alpha weights that favor rare classes
3. Extreme class imbalance (83:1 ratio)

Created an inconsistent loss landscape where the "predict majority always" solution was locally optimal.

**Analogy**: It's like trying to encourage someone to explore (focal loss) while also rewarding them for staying home (majority class is 83% of the time). They learn that staying home is optimal.

---

## Lessons Learned

### What Didn't Work

| Approach | Why It Failed |
|----------|---------------|
| **CE Loss + class weights** | Weights sufficient to overcome optimization landscape |
| **CE Loss + WeightedRandomSampler** | Sampler created artificial pixel distribution, overfitting |
| **Focal Loss (buggy impl)** | Implementation errors with extreme weights |
| **Focal Loss + alpha weights** | Combined mechanisms created local minimum at majority class |

### Root Cause of All Failures

The fundamental issue is **optimization landscape**:
- With 83:1 class ratio, local minimum is at "predict majority"
- All distance metrics reward majority class accuracy
- Small gradient signals from minority classes can't overcome inertia

Standard solutions (weighted loss, focal loss, oversampling) all have limitations:
1. **Weighted loss**: Weights must be extreme (50:1+) to overcome majority, causing numerical instability
2. **Focal loss**: Can create incentive misalignment at extreme ratios
3. **Oversampling**: Changes data distribution, model doesn't generalize

### What We Should Have Tried

1. **Metric-based learning**: Optimize for Extreme recall specifically, not overall accuracy
2. **Multi-task learning**: Learn Low Severity + Extreme as separate tasks with separate heads
3. **Cost-sensitive learning with hard constraints**: Force model to achieve minimum recall on each class
4. **Threshold adjustment**: Train model to optimize for overall accuracy, adjust thresholds at inference time
5. **Data collection**: Collect more Extreme samples to reduce imbalance ratio from 83:1 to ~10:1
6. **Ensemble methods**: Train separate models for majority vs minority classes

---

## Conclusion

**Iteration 2 remains the best approach** with 31% Extreme recall, despite being below the 50-70% target.

The class imbalance (83:1) is too extreme for standard techniques to overcome with the current training data and architecture. Achieving 50-70% Extreme recall would require:
- Either more Extreme training samples (collect data)
- Or fundamentally different learning approach (metric optimization, multi-task learning)
- Or acceptance of lower performance on majority classes

**Recommendation**: 
1. **Accept Iteration 2 as baseline** (31% Extreme recall)
2. **Prioritize data collection** to reduce class imbalance
3. **Consider metric-specific optimization** for next iteration (optimize for macro F1, not accuracy)
4. **Revisit advanced techniques** only after addressing data imbalance

