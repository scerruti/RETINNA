# Device Handling: CPU/GPU Compatibility

Pattern for running notebooks on CPU or GPU machines, following PA3 conventions.

## Quick Start

```python
from src.device_utils import get_device, move_to_device

# At start of notebook
device = get_device()  # Prints: "Using device: GPU_NAME" or "CPU"

# Load model
model = UNet(...).to(device)

# In training/inference loop
for batch in dataloader:
    batch = move_to_device(batch, device)  # Move all tensors to device
    
    images = batch['image']  # Now on GPU/CPU
    masks = batch['mask']    # Now on GPU/CPU
    
    # Forward pass
    outputs = model(images)
```

## Pattern Details

### 1. Initialize Device (Once at start)

```python
device = get_device()
# Output: "Using device: NVIDIA A100 GPU" (or "CPU")
```

**Returns**: `torch.device('cuda')` or `torch.device('cpu')`

### 2. Move Model to Device

```python
model = model.to(device)
```

**Important**: Do this ONCE after creating the model, not every iteration.

### 3. Move Batch Data to Device (In loop)

**Option A: Single tensors**
```python
images = images.to(device)
masks = masks.to(device)
```

**Option B: Entire batch dict (Recommended for DataLoaders)**
```python
batch = move_to_device(batch, device)
images = batch['image']
masks = batch['mask']
```

**Why**: CaBuAr DataLoader returns dict with 'image' and 'mask' keys. `move_to_device()` handles nested structures automatically.

### 4. Move Predictions Back to CPU (for visualization)

```python
predictions = predictions.cpu().numpy()
```

**Why**: Matplotlib and NumPy require CPU tensors or arrays.

## Template for Training Notebook

```python
import torch
from src.device_utils import get_device, move_to_device
from src.unet import UNet
from src.dataset import get_dataloaders

# 1. Setup device
device = get_device()

# 2. Load model
model = UNet(in_channels=12, out_channels=2).to(device)

# 3. Load dataloaders
dataloaders = get_dataloaders(batch_size=32)

# 4. Training loop
for epoch in range(num_epochs):
    for batch in dataloaders['train']:
        # Move batch to device
        batch = move_to_device(batch, device)
        
        # Forward pass
        images = batch['image']  # [B, 12, 512, 512] on device
        masks = batch['mask']    # [B, 1, 512, 512] on device
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Important Notes

1. **Model to device ONCE**: `model.to(device)` happens once after creation
2. **Batch to device EVERY ITERATION**: Happens inside the training loop
3. **Device-agnostic code**: Always use variables, never hardcode 'cuda' or 'cpu'
4. **Forward pass**: Model forward automatically computes on same device as inputs
5. **Visualization**: Move tensors back to CPU before plotting: `tensor.cpu()`

## PA3 Pattern Reference

| PA3 Code | Our Code |
|----------|----------|
| `device = torch.device(...)` | `device = get_device()` |
| Direct tensor move | `move_to_device(batch, device)` |
| Model training on GPU | `model.to(device)` |

## Debugging

If you get errors like "Tensor is on CUDA but X is on CPU":
1. Check model is on device: `print(next(model.parameters()).device)`
2. Check batch is on device: `print(batch['image'].device)`
3. Use `move_to_device()` to ensure everything matches

---

**Files**:
- `src/device_utils.py` — Reusable device utilities
- Notebooks: Import and use `get_device()` at start
