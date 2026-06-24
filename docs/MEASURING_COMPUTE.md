# Measuring Compute: GPU & Budget Tracking Guide

**Purpose**: Track actual compute usage during training experiments to stay within budget and optimize resource allocation.

---

## What to Measure

### 1. Colab Pro Dashboard Metrics

**Available in Colab Settings → Resources:**

```
Available compute units:     [Current balance]
Usage rate:                  [Units per hour]
Active sessions:             [Number of notebooks]

System RAM:                  [Used / Total] GB
GPU RAM:                     [Used / Total] GB  ← Most important
Disk:                        [Used / Total] GB
```

**What it means:**
- **Compute units**: Currency for GPU/TPU usage (1 unit ≈ 1 GPU hour on T4)
- **Usage rate**: How fast units are being consumed (varies by GPU type)
- **GPU RAM**: VRAM allocated to your kernel (peaks during forward/backward pass)
- **System RAM**: CPU memory (usually not limiting factor for training)

### 2. Training-Specific Metrics

**During training loop, track:**

```python
# At start of training
import torch
import time

start_time = time.time()
initial_gpu_memory = torch.cuda.memory_allocated() / 1e9  # Convert to GB

print(f"Initial GPU Memory: {initial_gpu_memory:.2f} GB")
print(f"Start Time: {start_time}")

# During training (per epoch)
for epoch in range(num_epochs):
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"Epoch {epoch}: Peak GPU Memory: {peak_memory:.2f} GB")
    torch.cuda.reset_peak_memory_stats()  # Reset for next epoch

# After training
end_time = time.time()
total_time = (end_time - start_time) / 60  # Convert to minutes
print(f"Total Training Time: {total_time:.2f} minutes")
```

### 3. Pre/Post Training Snapshots

**Capture before and after each experiment:**

| Metric | Pre-Training | Post-Training | Difference |
|--------|--------------|---------------|------------|
| Compute Units Available | 93.09 | ? | ? |
| GPU RAM Used | 0.0/15.0 GB | ? | ? |
| System RAM Used | 1.5/51.0 GB | ? | ? |
| Active Sessions | 1 | ? | ? |

---

## How to Track During Training

### Method 1: Manual Snapshots (Simple)

**Before starting training:**
1. Open Colab → Settings → Resources
2. Take screenshot or note values
3. Note wall-clock time

**After training completes:**
1. Open Colab → Settings → Resources (again)
2. Compare values
3. Calculate elapsed time

**Calculation:**
```
Compute units used = (Initial units - Final units)
GPU time estimate = Compute units used / Usage rate
```

### Method 2: Automated Logging (Better)

**Add to your training notebook:**

```python
import torch
import time
from datetime import datetime

class ComputeTracker:
    def __init__(self, experiment_name):
        self.name = experiment_name
        self.start_time = time.time()
        self.start_gpu_mem = torch.cuda.memory_allocated() / 1e9
        self.peak_gpu_mem = 0.0
        
    def log_epoch(self, epoch):
        current_mem = torch.cuda.memory_allocated() / 1e9
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        self.peak_gpu_mem = max(self.peak_gpu_mem, peak_mem)
        elapsed = (time.time() - self.start_time) / 60  # Minutes
        print(f"[Epoch {epoch}] GPU: {current_mem:.2f}GB peak: {peak_mem:.2f}GB elapsed: {elapsed:.1f}min")
        torch.cuda.reset_peak_memory_stats()
    
    def finish(self):
        elapsed_minutes = (time.time() - self.start_time) / 60
        elapsed_hours = elapsed_minutes / 60
        gpu_mem_delta = torch.cuda.memory_allocated() / 1e9 - self.start_gpu_mem
        
        print(f"\n{'='*70}")
        print(f"Experiment: {self.name}")
        print(f"Total time: {elapsed_minutes:.1f} min ({elapsed_hours:.2f} hours)")
        print(f"Peak GPU memory: {self.peak_gpu_mem:.2f} / 15.0 GB")
        print(f"GPU memory delta: {gpu_mem_delta:+.2f} GB")
        print(f"{'='*70}")

# Usage in training
tracker = ComputeTracker("Class Weighting (pos_weight=1.5)")

for epoch in range(num_epochs):
    train_loss = train_epoch(...)
    val_loss, val_iou = validate(...)
    tracker.log_epoch(epoch)

tracker.finish()
```

### Method 3: Manual Colab Pro Monitoring (Most Accurate)

**During training:**
1. Keep Colab → Settings → Resources panel open
2. Watch GPU RAM usage in real-time
3. Note peak usage
4. Watch usage rate (units/hour)

**What to observe:**
- GPU RAM starts at 0, rises during forward pass, peaks during backward pass
- Typical pattern: ~8-10 GB for 24-channel U-Net with batch size 4
- Usage rate changes depending on compute type (T4 vs A100)

---

## Interpretation Guide

### GPU Memory Usage

**Typical breakdown for U-Net training:**
```
Model weights:        ~120 MB (31M params × 4 bytes/param)
Batch data:          ~1.2 GB (4 × 2 × 12 × 512 × 512 × 4 bytes)
Activations (FW):    ~2 GB (intermediate feature maps)
Gradients (BW):      ~2 GB (backprop computation)
Optimizer state:     ~240 MB (Adam maintains 2 copies per param)
────────────────────────────
Peak during epoch:    ~8-10 GB
```

**Safe thresholds:**
- ✓ < 10 GB: Normal, no issues
- ⚠️ 10-12 GB: Getting tight, monitor closely
- ❌ > 12 GB: Risk of OOM error, reduce batch size

### Compute Units per Experiment

**Colab Pro rate (~$0.10 per unit):**
```
T4 GPU:       1 unit/hour
P100 GPU:     2.5 units/hour
A100 GPU:     8 units/hour

Example:
- 20 epochs on T4: ~20 min = 0.33 hours = 0.33 units (~$0.03)
- 50 epochs on T4: ~50 min = 0.83 hours = 0.83 units (~$0.08)
```

**Monthly budget tracking:**
```
Month quota:          ~1500 units
Day 3 baseline:       3 units (0.2%)
Day 4 tuning (est):   5-10 units (0.3-0.7%)
Remaining comfort:    ~1480 units (98.7% for rest of month)
```

### Time Conversion

```
GPU hours → Minutes × 60
Minutes → GPU hours ÷ 60

Example:
- 20 epochs, 20 min/epoch = 400 minutes = 6.67 hours
- But GPU time (units) depends on GPU type:
  - T4: 6.67 units
  - A100: 53 units (8× more expensive)
```

---

## Day 4 Sprint Tracking Template

**For Issue #14 (Class Weighting Experiment):**

```markdown
## Experiment: pos_weight=1.5 (20 epochs)

### Pre-Training
- Time: 12:46 PM
- Compute units: 93.09
- GPU RAM: 0.0 / 15.0 GB
- Usage rate: 1.27 units/hour

### During Training
- [Epoch 1] 0.45 GB peak
- [Epoch 5] 0.87 GB peak
- [Epoch 10] 1.20 GB peak
- [Epoch 15] 1.15 GB peak
- [Epoch 20] 1.18 GB peak → Best model

### Post-Training
- Time: 1:15 PM (estimated)
- Elapsed: 29 minutes
- Compute units used: ~0.40 units
- Compute units remaining: 92.7
- Peak GPU RAM: 1.20 GB (vs 15.0 GB available)

### Results
- Best Val IoU: ? (expected: 0.52+)
- Best Epoch: ?
- Final Recall: ? (expected: 65-70%)
- Final Precision: ? (expected: 91-93%)

### Conclusion
✓ Low GPU/compute usage
✓ Budget remaining for next experiment
→ Ready for Issue #15 (Loss Tuning)
```

---

## Common Issues & Fixes

### "I don't see compute units in settings"

**Solution:** Make sure you're in Colab Pro (paid subscription)
- Free Colab: No compute unit tracking
- Colab Pro: Shows in Settings → Resources

### "GPU RAM suddenly spikes to 14 GB during training"

**Likely cause:** Validation dataset loaded entirely into GPU  
**Fix:** Add `torch.cuda.empty_cache()` between training and validation:
```python
train_loss = train_epoch(...)
torch.cuda.empty_cache()  # Clear GPU memory
val_loss = validate(...)
```

### "Usage rate shows 1.27, but I ran for 30 min and lost 0.83 units"

**This is normal!** Usage rate fluctuates:
- Peak rate during compute-heavy operations
- Lower rate during I/O or idle
- Average over session = 0.83 units / 30 min ≈ 1.66 units/hour

---

## Best Practices

1. **Capture baseline before every experiment**
   - Screenshot or note Colab Pro dashboard
   - Timestamp it

2. **Log training progress**
   - Record per-epoch GPU memory
   - Track wall-clock time
   - Compare to actual compute units consumed

3. **Review post-training**
   - Calculate actual GPU time used
   - Compare to estimate
   - Use for budgeting future experiments

4. **Monitor monthly budget**
   - Track cumulative units used
   - Adjust experiment size if approaching limits
   - Plan ambitious experiments during high-quota periods

5. **Document in DAY_4_GPU_TRACKING.md**
   - Fill in actual metrics after each run
   - Update budget forecast
   - Inform next day's planning

---

## References

- [Colab Pro Documentation](https://colab.research.google.com)
- PyTorch CUDA Memory Management: `torch.cuda.memory_allocated()`, `torch.cuda.max_memory_allocated()`
- Typical T4 GPU memory: 15 GB VRAM

---

**Last Updated**: 2026-06-24  
**Related Issues**: #14 (Hyperparameter Tuning), #15 (Loss Optimization)  
**Related Docs**: DAY_4_GPU_TRACKING.md, DAY_4_SPRINT_PLAN.md
