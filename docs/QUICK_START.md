# Quick Start: Running RETINNA on Google Colab

Complete workflow to train and evaluate a burn scar segmentation model on Colab.

## Prerequisites

- Google account (for Colab and Google Drive)
- GPU quota on Colab (free tier includes limited GPU hours)

## Step-by-Step Workflow

### 1. Open Colab and Run Notebook 01

Go to [Google Colab](https://colab.research.google.com) and upload `notebooks/01_data_pipeline.ipynb`:

**File → Upload notebook → Choose `01_data_pipeline.ipynb`**

This notebook:
- Clones the RETINNA repository
- Installs dependencies (torch, torchgeo, etc.)
- Mounts Google Drive for dataset caching
- Loads and validates the CaBuAr dataset

**Expected output:**
```
✓ Google Drive caching enabled
✓ Train samples: 3098
✓ Val samples: 644
✓ Test samples: 644
```

### 2. Run Notebook 02 (Optional: Exploratory Analysis)

Upload `notebooks/02_exploratory_analysis.ipynb` to the same Colab session.

This notebook explores:
- Per-band statistics (pre/post-fire)
- NDVI analysis for vegetation change
- Spectral signatures of burned vs unburned areas
- Data quality checks

**Skip if you just want to train.**

### 3. Run Notebook 03 (Training)

Upload `notebooks/03_training.ipynb` to Colab.

**On first run:**
1. Execute all cells sequentially
2. Use the epochs slider (default 5, adjust 1-50)
3. Training runs on GPU if available (Tesla T4 on free tier)

**Expected output:**
```
Epoch 1/5
Train Loss: 0.6234
Val Loss: 0.5891, Val IoU: 0.0245
★ Best model saved

[Training curves plot]
```

**Training time (T4 GPU):**
- 5 epochs: ~5 minutes
- 20 epochs: ~20 minutes
- 50 epochs: ~50 minutes

**Model checkpoint saved to:** `checkpoints_notebook/best_model.pth`

### 4. Run Notebook 04 (Inference)

Upload `notebooks/04_inference.ipynb` to Colab.

This notebook:
- Loads the trained model checkpoint
- Runs inference on full test set
- Generates predictions for all test samples
- Saves results to `inference_results/predictions.pt`
- Shows sample predictions vs ground truth

**Expected output:**
```
✓ Inference complete
  Predictions shape: [644, 512, 512]
  Targets shape: [644, 512, 512]
✓ Saved predictions to inference_results/predictions.pt

[Sample prediction visualizations]
```

### 5. Run Notebook 05 (Analysis & Metrics)

Upload `notebooks/05_analysis.ipynb` to Colab.

This notebook loads the predictions from 04 and computes:
- **IoU** (Intersection over Union) for burned class
- **Precision, Recall, F1-score**
- **Confusion matrix**
- **Error analysis and visualizations**
- **Baseline report**

**Expected output:**
```
Burned Class Metrics:
  IoU: 0.68
  Precision: 0.71
  Recall: 0.65
  F1-Score: 0.68

[Confusion matrix and detailed visualizations]
```

## Training Tips

### If Running Out of Memory
In notebook 03, reduce batch size:
```python
dataloaders = get_dataloaders(
    batch_size=2,  # Reduce from 4
    ...
)
```

### If Training Disconnects
The Colab kernel continues running even if your browser disconnects. You can:
1. Refresh browser and reconnect
2. Check if checkpoint exists: `checkpoints_notebook/best_model.pth`
3. Resume from checkpoint (if supported by notebook)

### GPU Not Available
If you get "CPU" device message:
- Free tier Colab may run CPU-only during peak hours
- Try again later
- Or run with fewer epochs on CPU (slower but works)

## Full Training Run (Recommended)

For a proper baseline model:

1. **Run 03** with **20 epochs** (15-20 min on GPU)
2. **Run 04** for inference (10 min)
3. **Run 05** for detailed metrics (5 min)

Total: ~45 minutes of GPU time

## File Organization

During a full Colab run, these directories are created:

```
/content/RETINNA/
├── checkpoints_notebook/          # Training checkpoints
│   ├── best_model.pth             # Best model weights
│   └── config.json                # Training config
│
├── inference_results/             # Predictions from 04
│   └── predictions.pt             # Predictions tensor
│
├── notebooks/                     # All 5 notebooks
│   ├── 01_data_pipeline.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_training.ipynb
│   ├── 04_inference.ipynb
│   └── 05_analysis.ipynb
│
└── src/                           # Core modules
    ├── unet.py                    # U-Net model
    ├── dataset.py                 # Dataset loaders
    ├── colab_utils.py             # Drive caching
    └── device_utils.py            # GPU/CPU handling
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torchgeo'"
- Run the pip install cell in setup
- Restart kernel and re-run

### "Transport endpoint is not connected" (Google Drive)
- First time: Normal, dataset is downloading
- Subsequent times: Clear Google Drive cache and retry
  ```python
  import shutil
  shutil.rmtree('/content/drive/MyDrive/RETINNA_DATA')
  ```

### Training very slow on CPU
- This is normal for 512×512 images and 31M parameter model
- Switch to GPU if possible
- Reduce batch size or epochs for testing

### Model checkpoint not found in 04
- Make sure you ran and completed 03 first
- Checkpoint saves to `checkpoints_notebook/best_model.pth`

## Documentation

For deeper understanding:
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) — File organization and purposes
- [TRAINING_PROCESS.md](TRAINING_PROCESS.md) — Training loop, loss functions, hyperparameters
- [DEVICE_HANDLING.md](DEVICE_HANDLING.md) — CPU/GPU compatibility
- [ARCHITECTURE_RATIONALE.md](ARCHITECTURE_RATIONALE.md) — Why U-Net, not FCN

## Next Steps After Baseline

Once you have a working baseline (20 epochs, ~0.65-0.70 IoU):

1. **Hyperparameter Tuning** — Adjust learning rate, epochs, batch size
2. **Multi-class Training** — Extend to burn severity levels
3. **Cross-satellite Transfer** — Test on Landsat 8 imagery
4. **Production Deployment** — Export model for inference on new imagery

---

**Estimated time for full workflow:** 1-2 hours (depending on epochs trained)
