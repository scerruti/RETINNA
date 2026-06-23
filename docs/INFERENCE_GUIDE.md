# Inference Guide: Using RETINNA for Burn Scar Prediction

How to run inference on new Sentinel-2 satellite imagery to generate burn scar predictions.

## Overview

Once you have a trained model checkpoint, you can:
1. **Generate predictions** on any bi-temporal Sentinel-2 imagery
2. **Quantify burn damage** (acreage calculations)
3. **Visualize fire perimeters** with pixel-level precision
4. **Compare predictions** to official CAL FIRE boundaries

## Quick Inference (Colab)

### Method 1: Interactive Notebook (Recommended)

Use **`notebooks/04_inference.ipynb`** on Colab:

```python
# 1. Load your trained model
checkpoint_path = Path('checkpoints_notebook/best_model.pth')
model = UNet(in_channels=24, out_channels=2).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# 2. Load your imagery (bi-temporal Sentinel-2)
# Must be shape: [B, 2, 12, 512, 512]
# - B: batch (number of tiles)
# - 2: timesteps (pre-fire, post-fire)
# - 12: spectral bands
# - 512: height/width

# 3. Run inference
with torch.no_grad():
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    burned_prob = probs[:, 1]  # Probability of burned class
    
# 4. Generate binary mask (threshold 0.5)
burned_mask = (burned_prob > 0.5).float()
```

### Method 2: Python Script

```python
import torch
from src.unet import UNet
from src.device_utils import get_device

# Setup
device = get_device()
model = UNet(in_channels=24, out_channels=2).to(device)
model.load_state_dict(torch.load('checkpoints_notebook/best_model.pth', map_location=device))
model.eval()

# Load your bi-temporal imagery
# Shape: [1, 2, 12, 512, 512] for single image pair
images = torch.randn(1, 2, 12, 512, 512).to(device)

# Inference
with torch.no_grad():
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    burned_prob = probs[:, 1]  # [1, 512, 512]

# Save prediction
torch.save(burned_prob.cpu(), 'prediction.pt')
```

## Input Format

### Bi-temporal Sentinel-2 Data

RETINNA expects **pre- and post-fire Sentinel-2 imagery** in this exact format:

**Shape:** `[batch_size, 2_timesteps, 12_bands, 512, 512]`

**Bands** (12 total):
```
Index  Band Name              Resolution  Purpose
-----  --------------------  ----------  ----------
0      Coastal Aerosol       60m         Atmospheric
1      Blue                  10m         Visible
2      Green                 10m         Visible
3      Red                   10m         Visible
4      Vegetation Red Edge   20m         Plant stress
5      Vegetation Red Edge   20m         Plant stress
6      Vegetation Red Edge   20m         Plant stress
7      NIR (Near-IR)         10m         *** Key for fire detection
8      Narrow NIR            20m         Plant vigor
9      Water Vapor           60m         Moisture
10     SWIR (Short-wave IR)  20m         *** Key for fire detection
11     Cloud Probability     20m         Data quality
```

**Key bands for burn detection:**
- **Band 7 (NIR)**: Healthy vegetation has high reflectance; burned areas are dark
- **Band 10 (SWIR)**: Burned areas have high reflectance; vegetation is dark

### Data Preparation

If you have raw GeoTIFFs:

```python
import rasterio
import numpy as np

# Read pre-fire image
with rasterio.open('pre_fire.tif') as src:
    pre_fire = src.read()  # [12, 512, 512]

# Read post-fire image
with rasterio.open('post_fire.tif') as src:
    post_fire = src.read()  # [12, 512, 512]

# Stack into bi-temporal format
images = np.stack([pre_fire, post_fire], axis=0)  # [2, 12, 512, 512]
images = np.expand_dims(images, axis=0)  # [1, 2, 12, 512, 512]

# Convert to tensor
images = torch.from_numpy(images).float()

# Normalize (Sentinel-2 standard: ÷10000)
images = images / 10000.0  # Maps to [0, 1] range
```

### Using CaBuAr Dataset (Testing)

If testing with CaBuAr dataset:

```python
from src.dataset import get_dataloaders

dataloaders = get_dataloaders(batch_size=4, normalize=True)

# Get a batch of test data
test_loader = dataloaders['test']
batch = next(iter(test_loader))

images = batch['image']  # [4, 2, 12, 512, 512]
masks = batch['mask']    # [4, 1, 512, 512] - ground truth

# Run inference
with torch.no_grad():
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    predictions = probs[:, 1]  # [4, 512, 512]
```

## Output Interpretation

### Prediction Outputs

**Raw model output:** `outputs` shape [B, 2, 512, 512]
- Channel 0: Logits for unburned class
- Channel 1: Logits for burned class

**Softmax probabilities:** `probs` shape [B, 2, 512, 512]
- `probs[:, 0]`: Probability of unburned (0-1)
- `probs[:, 1]`: Probability of burned (0-1)
- Sum to 1.0 across channels

**Binary mask:** `burned_mask` shape [B, 512, 512]
- 1.0: Predicted burned
- 0.0: Predicted unburned
- Threshold: 0.5 (customizable)

### Interpreting Predictions

```python
# Probability-based (soft predictions)
burned_prob = probs[:, 1]  # [0, 1] - confidence in burn

# Binary mask (hard predictions)
burned_mask = (burned_prob > 0.5).float()

# Confidence regions (thresholding)
high_confidence_burned = burned_prob > 0.8
uncertain_region = (burned_prob > 0.2) & (burned_prob < 0.8)
high_confidence_unburned = burned_prob < 0.2

# Visualize probabilities
import matplotlib.pyplot as plt
plt.imshow(burned_prob[0].cpu().numpy(), cmap='YlOrRd', vmin=0, vmax=1)
plt.colorbar(label='Burn Probability')
plt.show()
```

## Computing Burned Area

### Pixel-level Calculation

```python
import numpy as np

# Get binary prediction
burned_mask = (burned_prob > 0.5).float()  # [B, 512, 512]

# Sentinel-2 resolution: 10m per pixel
resolution_m = 10

for b in range(burned_mask.shape[0]):
    num_burned_pixels = burned_mask[b].sum().item()
    burned_area_m2 = num_burned_pixels * (resolution_m ** 2)
    burned_area_hectares = burned_area_m2 / 10000
    burned_area_acres = burned_area_hectares * 2.471
    
    print(f"Tile {b}:")
    print(f"  Burned pixels: {num_burned_pixels}")
    print(f"  Area: {burned_area_hectares:.2f} ha ({burned_area_acres:.2f} acres)")
```

### Comparing to Ground Truth

```python
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Get predictions and ground truth
pred_binary = (predictions > 0.5).cpu().numpy().flatten()
true_binary = masks.squeeze(1).cpu().numpy().flatten()

# Metrics
precision, recall, f1, _ = precision_recall_fscore_support(true_binary, pred_binary, average='binary')
cm = confusion_matrix(true_binary, pred_binary)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
print(f"Confusion Matrix:\n{cm}")
```

## Advanced: Batch Inference

### Process Multiple Images

```python
# Stack multiple image pairs
all_images = []
for pre_fire_path, post_fire_path in image_pairs:
    # Load and preprocess each pair
    pre = load_and_normalize(pre_fire_path)   # [12, 512, 512]
    post = load_and_normalize(post_fire_path)  # [12, 512, 512]
    stacked = np.stack([pre, post], axis=0)    # [2, 12, 512, 512]
    all_images.append(stacked)

# Batch process
images_batch = np.stack(all_images)  # [N, 2, 12, 512, 512]
images_tensor = torch.from_numpy(images_batch).float().to(device)

with torch.no_grad():
    outputs = model(images_tensor)
    predictions = torch.softmax(outputs, dim=1)[:, 1]  # [N, 512, 512]

# Save all predictions
torch.save(predictions.cpu(), 'batch_predictions.pt')
```

### GPU Memory Management

For large batches:

```python
batch_size = 4
total_samples = len(image_pairs)

all_predictions = []

for i in range(0, total_samples, batch_size):
    batch = images[i:i+batch_size].to(device)
    
    with torch.no_grad():
        outputs = model(batch)
        preds = torch.softmax(outputs, dim=1)[:, 1]
    
    all_predictions.append(preds.cpu())

# Concatenate all predictions
all_predictions = torch.cat(all_predictions, dim=0)
```

## Troubleshooting

### Shape Mismatch Error

```
ValueError: Expected input to have 24 channels, got 12
```

**Fix:** Flatten timesteps into channels
```python
B, T, C, H, W = images.shape
images_flat = images.view(B, T * C, H, W)  # [B, 24, 512, 512]
```

### Low Prediction Quality

**Possible causes:**
1. **Normalization mismatch** - Ensure images are divided by 10000 (0-1 range)
2. **Different spatial resolution** - Model expects 512×512
3. **Different spectral bands** - Must have exactly 12 bands
4. **Temporal misalignment** - Pre and post-fire must be aligned
5. **Low model performance** - Baseline may need improvement

**Debug:**
```python
print(f"Image range: {images.min():.3f} to {images.max():.3f}")  # Should be [0, 1]
print(f"Image shape: {images.shape}")  # Should be [B, 24, 512, 512]
print(f"Prediction range: {predictions.min():.3f} to {predictions.max():.3f}")  # Should be [0, 1]
```

### Memory Issues on GPU

**Solutions:**
1. Reduce batch size: `batch_size = 2` or `1`
2. Process sequentially instead of batched
3. Free memory: `torch.cuda.empty_cache()`

```python
# Process one at a time
for i, image_pair in enumerate(image_pairs):
    image_tensor = torch.tensor(image_pair).unsqueeze(0).to(device)
    pred = model(image_tensor)
    torch.cuda.empty_cache()
```

## Performance Expectations

| Dataset | IoU | Precision | Recall | F1-Score |
|---------|-----|-----------|--------|----------|
| CaBuAr (baseline) | 0.65-0.70 | 0.68-0.72 | 0.62-0.68 | 0.65-0.70 |
| After tuning | 0.72-0.78 | 0.74-0.80 | 0.70-0.76 | 0.72-0.78 |

**Inference speed (T4 GPU):**
- Single 512×512 tile: ~50-100 ms
- Batch of 32 tiles: ~1.5-2 seconds
- 1000 tiles: ~1-2 minutes

## Integration with External Tools

### Convert to GeoTIFF

```python
import rasterio
from rasterio.transform import Affine

# Get prediction
burned_prob = predictions[0].cpu().numpy()

# Define geotransform (example - adjust for your imagery)
# Transform: (x_offset, pixel_width, 0, y_offset, 0, -pixel_height)
transform = Affine.identity()

# Write GeoTIFF
with rasterio.open(
    'output_prediction.tif',
    'w',
    driver='GTiff',
    height=512, width=512,
    count=1,
    dtype=burned_prob.dtype,
    transform=transform
) as dst:
    dst.write(burned_prob, 1)
```

### Send to QGIS for Mapping

1. Export predictions as GeoTIFF (above)
2. Open in QGIS
3. Overlay with original satellite image
4. Compare with CAL FIRE fire perimeter shapefiles

---

See [QUICK_START.md](QUICK_START.md) for complete workflow.
