# Training Process: U-Net Burn Detection

Comprehensive guide to training the U-Net model on CaBuAr dataset using BCE+Dice loss for binary burn scar detection.

## Overview

The training pipeline implements a standard supervised learning loop following the PA3 pattern:
1. Load bi-temporal Sentinel-2 imagery and ground truth burn masks
2. Forward pass through U-Net (24-channel input → 2-class output)
3. Compute hybrid BCE+Dice loss
4. Backpropagation and optimization
5. Validation with IoU metric
6. Checkpoint management and early stopping

## Loss Function: BCE+Dice Hybrid

### Why Two Losses?

**Class Imbalance Problem:**
- CaBuAr dataset is ~90% unburned pixels, ~10% burned
- Standard cross-entropy loss biases toward the majority class
- Model learns to predict "unburned" everywhere and ignores burn scars

**Solution:**
- **BCE (Binary Cross-Entropy):** Penalizes incorrect predictions at per-pixel level
- **Dice Loss:** Emphasizes recall/precision balance, less sensitive to class imbalance

### Mathematical Definition

```
L_total = w_bce * L_bce + w_dice * L_dice

where:
  L_bce  = BCEWithLogitsLoss(burned_logits, targets)
  L_dice = 1 - (2 * TP + ε) / (TP + FP + FN + ε)
  
  TP   = Σ(predicted_burned * targets)
  FP   = Σ(predicted_burned) - TP
  FN   = Σ(targets) - TP
  ε    = 1e-8 (numerical stability constant)
  
  w_bce  = 0.5  (default weight)
  w_dice = 0.5  (default weight)
```

### Implementation Details

**Input/Output Shapes:**
```python
predictions: [B, 2, H, W]  # Raw logits (unburned, burned classes)
targets:     [B, 1, H, W]  # Binary mask (0=unburned, 1=burned)
```

**Step 1: BCE Loss**
- Extract burned class logits: `predictions[:, 1:2]` → [B, 1, H, W]
- Apply BCEWithLogitsLoss (sigmoid applied internally)
- Treats each pixel independently as binary classification

**Step 2: Dice Loss**
- Softmax on all classes to get probabilities: [B, 2, H, W]
- Extract burned probability: `softmax_output[:, 1:2]` → [B, 1, H, W]
- Compute intersection/union with ground truth
- Dice coefficient ranges [0, 1], loss = 1 - Dice

**Step 3: Combine**
- Equally weight both (0.5 each by default)
- Empirically validated on class imbalance problems

## Training Loop

### Command Line Usage

```bash
# Smoke test (1 epoch, small batch)
python train.py --epochs 1 --batch-size 2 --learning-rate 0.0005

# Full training (50 epochs, standard config)
python train.py --epochs 50 --batch-size 32 --learning-rate 0.0005

# Resume from checkpoint
python train.py --epochs 50 --resume checkpoints/best.pth

# With early stopping
python train.py --epochs 100 --early-stopping --early-stopping-patience 10
```

### Arguments

| Argument | Default | Range | Notes |
|----------|---------|-------|-------|
| `--epochs` | 50 | 1-500 | Total training epochs |
| `--batch-size` | 32 | 2-64 | Reduce to 16/8 if OOM with 12 channels |
| `--learning-rate` | 0.0005 | 0.0001-0.001 | Adam optimizer (not SGD) |
| `--num-workers` | 0 | 0-8 | Data loading parallelism |
| `--checkpoint-dir` | checkpoints/ | any path | Where to save models |
| `--resume` | None | path | Resume from checkpoint |
| `--early-stopping` | False | flag | Enable early stopping |
| `--early-stopping-patience` | 10 | 5-20 | Epochs without improvement before stop |

### Epoch Workflow

**Training Phase:**
1. Set model to train mode (enables dropout/batch norm updates)
2. For each batch:
   - Load images [B, 2, 12, 512, 512] and masks [B, 1, 512, 512]
   - Flatten timesteps: [B, 2, 12, 512, 512] → [B, 24, 512, 512]
   - Forward pass through U-Net
   - Compute BCE+Dice loss
   - Zero gradients, backpropagate, optimizer step
   - Log batch loss every 10 batches
3. Average training loss across all batches

**Validation Phase:**
1. Set model to eval mode (no dropout updates)
2. For each batch (no gradient computation):
   - Forward pass
   - Compute loss
   - Compute IoU metric (Intersection over Union for burned class)
3. Average loss and IoU across all batches
4. Log metrics and checkpoint if best so far

### Checkpointing

**Latest Checkpoint** (`latest.pth`)
- Saved every epoch
- Allows resuming mid-training
- Contains: model weights, optimizer state, epoch, validation IoU

**Best Checkpoint** (`best.pth`)
- Saved only when validation IoU improves
- Use this for inference (best model seen so far)
- Marked with ★ in logs

**Configuration** (`config.json`)
- Saved at training start
- Records all hyperparameters used
- Useful for reproducing results

### Early Stopping

If enabled:
- Monitor validation IoU
- Stop if no improvement for N epochs (patience)
- Prevents overfitting, saves computation
- Default: disabled (full training preferred for this task)

## Device Handling (CPU/GPU)

Training automatically:
- Detects GPU availability (`torch.cuda.is_available()`)
- Prints device name at start
- Moves model and batches to device
- Falls back to CPU on Colab if GPU unavailable

Optimal configurations:
- **GPU (A100/V100):** `--batch-size 32`, `--learning-rate 0.0005`
- **CPU (Colab):** `--batch-size 2`, `--learning-rate 0.0001` (reduce for stability)

## Google Drive Caching (Colab)

To avoid re-downloading the ~6GB dataset:

```python
from src.colab_utils import setup_cabuaur_cached

root = setup_cabuaur_cached()  # Uses Google Drive cache
```

- First run: downloads and caches to Drive
- Subsequent runs: loads from Drive (much faster)
- Automatic fallback to default cache if not on Colab

## Data Format

### Bi-temporal Structure

CaBuAr provides pre-fire and post-fire imagery:
- **Input:** [B, 2, 12, 512, 512] (batch, timesteps, bands, height, width)
- **Bands:** 11 Sentinel-2 bands + 1 cloud probability
- **Processing:** Flattened to [B, 24, 512, 512] for U-Net

### Normalization

Default: Divide by 10000 (Sentinel-2 L2A radiometric standard)
- Maps DN values 0-10000 to 0-1 range
- Enables consistent training across scenes
- Disable with `normalize=False` if using pre-normalized data

### Masks

Ground truth from CaBuAr:
- 0 = Unburned (healthy vegetation)
- 1 = Burned (fire-affected area)
- Binary segmentation task (not multi-class)

## Metrics

### Loss (Training & Validation)

Combined BCE+Dice loss value:
- Lower is better
- Typically ranges 0.1-0.5 for well-trained models
- Watch for loss plateauing or diverging

### IoU (Validation Only)

Intersection over Union for burned class:
```
IoU = |predicted ∩ ground_truth| / |predicted ∪ ground_truth|
```

- Ranges [0, 1] where 1 = perfect prediction
- Threshold: 0.5 (pixel predicted burned if P(burned) > 0.5)
- Targets:
  - Baseline: 0.40-0.50 (first epoch results)
  - Good: 0.65-0.75 (after training)
  - Strong: 0.80+ (with hyperparameter tuning)

## Hyperparameter Recommendations

### Learning Rate

Default 0.0005 works well, but tune if needed:
- **Too high (>0.001):** Loss diverges, NaN values
- **Too low (<0.0001):** Slow convergence, barely improving
- **Safe range:** 0.0001-0.001

### Batch Size

Limited by GPU/CPU memory:
- **GPU (A100):** 32-64 recommended
- **GPU (smaller):** 16-32
- **CPU (Colab):** 2-4 (slow but works)
- Larger batches = more stable gradients

### Epochs

50 is reasonable default:
- Early epochs: rapid improvement
- Plateauing after 30-40 epochs (diminishing returns)
- Early stopping recommended if loss/IoU stalls

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| OOM (out of memory) | Batch too large for device | Reduce `--batch-size` to 16/8/2 |
| Loss = NaN | Learning rate too high | Reduce `--learning-rate` to 0.0001 |
| Loss not improving | Underfitting or poor LR | Increase `--learning-rate` slightly or ensure data loads |
| GPU not detected | CUDA not available | Falls back to CPU automatically |
| Dataset download fails | Corrupted cache | Delete `~/.cache/torchgeo/` and retry |
| Colab dataset re-downloads | Not using Drive cache | Call `setup_cabuaur_cached()` in train.py |

## Files

- `train.py` — Main training script
- `src/unet.py` — U-Net model
- `src/dataset.py` — CaBuAr dataset wrapper
- `src/device_utils.py` — Device management
- `src/colab_utils.py` — Colab-specific utilities
- `checkpoints/` — Saved models and config
