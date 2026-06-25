# Loss Function Configuration Guide

## Quick Start

To use a different loss function in the II_02 notebook, modify `config.py` and restart the kernel. No need to regenerate the entire notebook.

### Step 1: Edit `config.py`

In the `notebooks/` directory, open `config.py` and change:

```python
LOSS_CONFIG = {
    'name': 'tversky',  # Change this to 'ce', 'dice', 'tversky', or 'focal'
    'params': {
        'alpha': 0.3,    # TverskyLoss parameter
        'beta': 0.7,     # TverskyLoss parameter
    },
    'use_class_weights': True,
}
```

### Step 2: In the Notebook

Add this cell after imports:

```python
# Import configurable losses
import sys
sys.path.append('/content/drive/MyDrive/RETINNA/notebooks')
from losses import get_loss
from config import LOSS_CONFIG

# Create loss function
print(f"Using loss: {LOSS_CONFIG['name']}")
print(f"Params: {LOSS_CONFIG['params']}")
criterion = get_loss(LOSS_CONFIG['name'], 
                     class_weights=class_weights if LOSS_CONFIG['use_class_weights'] else None,
                     **LOSS_CONFIG['params'])
criterion = criterion.to(device)
```

### Step 3: Restart and Train

- Restart kernel
- Run cells up to the training loop
- Run training loop
- Training uses the loss from config.py

---

## Loss Function Options

### 1. CrossEntropyLoss (CE)

**Configuration:**
```python
LOSS_CONFIG = {
    'name': 'ce',
    'params': {},
    'use_class_weights': True,
}
```

**When to use:**
- Baseline comparison
- When class balance is reasonable
- Standard per-pixel likelihood training

**Pros:**
- Numerically stable
- Well-understood behavior
- No hyperparameters to tune

**Cons:**
- Vulnerable to class imbalance
- Rewards predicting majority class

---

### 2. DiceLoss

**Configuration:**
```python
LOSS_CONFIG = {
    'name': 'dice',
    'params': {},
    'use_class_weights': True,
}
```

**When to use:**
- Want to optimize spatial overlap (IoU) directly
- Need balanced multi-class performance
- Prefer simpler loss function

**Pros:**
- Robust to class imbalance
- Directly optimizes intersection/union
- No hyperparameters to tune

**Cons:**
- Less sensitive to per-pixel calibration
- May underfit in early epochs

---

### 3. TverskyLoss (RECOMMENDED)

**Configuration:**
```python
LOSS_CONFIG = {
    'name': 'tversky',
    'params': {
        'alpha': 0.3,  # FP penalty (lower = accept more false positives)
        'beta': 0.7,   # FN penalty (higher = penalize false negatives more)
    },
    'use_class_weights': True,
}
```

**Key Insight:**
- `alpha=0.3, beta=0.7` penalizes **false negatives 2.3× more** than false positives
- This biases the model toward detecting Extreme pixels (high recall)
- Accepts some false positives in exchange for fewer missed Extreme pixels

**When to use:**
- **THIS IS RECOMMENDED FOR EXTREME CLASS** (currently 31% recall is too low)
- Need high recall on specific rare class
- Can trade some false positives for better detection

**Pros:**
- Directly optimizes for recall
- Smooth gradients
- Theoretically sound for imbalance

**Cons:**
- More hyperparameters to tune
- Need to balance alpha/beta for your use case

**Hyperparameter Tuning Guide:**
```
For HIGH RECALL (minimize false negatives):
  alpha < beta (penalize FN more)
  Example: alpha=0.2, beta=0.8 (4× more penalty on FN)

For BALANCED RECALL/PRECISION:
  alpha ≈ beta
  Example: alpha=0.5, beta=0.5 (equal penalties)

For HIGH PRECISION (minimize false positives):
  alpha > beta (penalize FP more)
  Example: alpha=0.8, beta=0.2 (4× more penalty on FP)
```

---

### 4. FocalLoss

**Configuration:**
```python
LOSS_CONFIG = {
    'name': 'focal',
    'params': {
        'gamma': 2.0,  # Focusing parameter
    },
    'use_class_weights': True,
}
```

**When to use:**
- Class imbalance is moderate (10:1 to 50:1)
- Want to focus on hard examples

**Pros:**
- Theoretically sound for handling imbalance
- Can work with aggressive focusing

**Cons:**
- **WARNING**: Failed catastrophically in Iteration 5 at 83:1 ratio
- Can create local minima where model predicts majority class
- Combine focal focusing + alpha weighting = inconsistent gradients

**Recommendation:**
- Only use if TverskyLoss fails
- Monitor training carefully for collapse

---

## Presets

Instead of editing `config.py` manually, use presets:

```python
from config import get_config_preset, LOSS_CONFIG

# Load a preset
LOSS_CONFIG = get_config_preset('iteration_6_tversky_recall')

# Available presets:
# - 'iteration_2_baseline': CE Loss (control)
# - 'iteration_5_focal': Focal Loss (comparison)
# - 'iteration_6_tversky_recall': Tversky optimized for high recall
# - 'iteration_6_tversky_balanced': Tversky balanced (alpha=beta=0.5)
# - 'iteration_6_dice': Dice Loss
```

---

## How to Test Experiments

### Experiment 1: Tversky vs CE (baseline)

**Run 1: Baseline**
```python
LOSS_CONFIG = get_config_preset('iteration_2_baseline')
```
- Train 20 epochs
- Record Extreme recall

**Run 2: Tversky Recall-Biased**
```python
LOSS_CONFIG = get_config_preset('iteration_6_tversky_recall')
```
- Train 20 epochs
- Compare Extreme recall to baseline
- Check if recall improved

### Experiment 2: Tune Tversky Alpha/Beta

```python
# Test 1: More aggressive FN penalty
LOSS_CONFIG['params'] = {'alpha': 0.2, 'beta': 0.8}

# Test 2: Even more aggressive
LOSS_CONFIG['params'] = {'alpha': 0.1, 'beta': 0.9}

# Test 3: Balanced
LOSS_CONFIG['params'] = {'alpha': 0.5, 'beta': 0.5}
```

For each, train and compare Extreme recall.

---

## Implementation Details

### What Each Loss Function Optimizes

```
CrossEntropyLoss:
  - Per-pixel likelihood: sum(-log(p_true_class))
  - Gradient: depends on confidence in predicted class
  - Vulnerable to majority class dominating gradients

DiceLoss:
  - Spatial overlap: (2*TP) / (FP + FN + 2*TP)
  - Gradient: depends on overall intersection/union
  - Balanced across classes naturally

TverskyLoss:
  - Tunable recall/precision: TP / (TP + alpha*FP + beta*FN)
  - Gradient: tunable via alpha/beta
  - Can bias toward recall (alpha < beta) or precision

FocalLoss:
  - Focused likelihood: -(1-p_t)^γ * log(p_t)
  - Gradient: downweights easy examples, focuses hard examples
  - Risk: Can create local minima at extreme imbalance
```

### Numerical Stability

All implementations:
- Use log_softmax for numerical stability (Focal, CE)
- Add eps=1e-6 for division by zero protection (Dice, Tversky)
- Avoid intermediate overflow by computing in log space where needed

---

## Next Steps

1. **Run Iteration 6 with TverskyLoss (recall-biased)**
   ```
   python_config.LOSS_CONFIG = get_config_preset('iteration_6_tversky_recall')
   ```
   Expected: Extreme recall improve from 31% toward 50-70%

2. **If TverskyLoss improves Extreme recall:**
   - Tune alpha/beta to optimize further
   - Experiment with different ratios

3. **If TverskyLoss still fails:**
   - Try DiceLoss (simpler alternative)
   - Consider data collection to reduce 83:1 imbalance

4. **Document results in ITERATION_ANALYSIS.md**
   - Add Iteration 6 results section
   - Compare to previous iterations

---

## Troubleshooting

**Q: How do I change the loss function?**
A: Edit `LOSS_CONFIG` in `config.py`, restart kernel, re-run training cell.

**Q: Can I change loss mid-training?**
A: No, you must restart the kernel. Loss function is fixed once training starts.

**Q: What if training diverges?**
A: 
- Check loss values are reasonable (not NaN or Inf)
- Try a different loss function
- Reduce learning rate
- Check class_weights are being applied

**Q: Which loss should I use?**
A: Start with `iteration_6_tversky_recall`. If it works, you're done. If not, try `iteration_6_dice`.

