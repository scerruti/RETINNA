# Quick Start: Running RETINNA on Google Colab

Complete workflow to train and evaluate a burn scar segmentation model on Colab.

## Prerequisites

- Google account (for Colab and Google Drive)
- GPU quota on Colab (free tier includes limited GPU hours)

## Step-by-Step Workflow

### 1. Open Colab and Run Notebook 01 — CPU

**Machine type:** CPU only (GPU not needed for data loading)  
**Estimated time:** 10 minutes  
**GPU hours saved:** ~10 min

Go to [Google Colab](https://colab.research.google.com):
1. Open `notebooks/01_data_pipeline.ipynb`
2. **Runtime → Change runtime type → CPU** (free tier default)
3. Run all cells

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

### 2. Run Notebook 02 (Optional: Exploratory Analysis) — CPU

**Machine type:** CPU only (data analysis, no training)  
**Estimated time:** 10 minutes  
**GPU hours saved:** ~10 min  
**Optional:** Skip if you just want to train.

Open `notebooks/02_exploratory_analysis.ipynb` in **same Colab session as 01** (or new session on CPU).

This notebook explores:
- Per-band statistics (pre/post-fire)
- NDVI analysis for vegetation change
- Spectral signatures of burned vs unburned areas
- Data quality checks

### 3. Run Notebook 03 (Training) — GPU Required

**Machine type:** GPU (Tesla T4 or better)  
**Estimated time:** 20 epochs ≈ 20 minutes (T4)  
**GPU hours required:** ~0.33 hours for 20 epochs  
**Session type:** New session (fresh GPU)

Open `notebooks/03_training.ipynb`:
1. **Runtime → Change runtime type → GPU**
2. Execute all cells sequentially
3. Use the epochs slider (default 5, adjust 1-50)
4. Model trains on T4 GPU

**Expected output:**
```
Epoch 1/20
Train Loss: 0.5694
Val Loss: 0.5742, Val IoU: 0.4177
★ Best model saved

[Training curves plot]
```

**Training time (Tesla T4 GPU):**
- 5 epochs: ~5 minutes
- 20 epochs: ~20 minutes (recommended baseline)
- 50 epochs: ~50 minutes

**Model checkpoint saved to:**
- Local: `checkpoints_notebook/best_model.pth`
- Drive backup: `/content/drive/MyDrive/RETINNA_checkpoints/best_model.pth` (auto-saved)

### 4. Run Notebook 04 (Inference) — GPU Recommended

**Machine type:** GPU (Tesla T4) or CPU (slower)  
**Estimated time:** 10 minutes (T4) or 30 minutes (CPU)  
**Session type:** New session with GPU, or same session as 03

Open `notebooks/04_inference.ipynb`:
1. **Runtime → Change runtime type → GPU** (recommended)
2. Run all cells
3. Automatically copies best_model.pth from Drive if needed

This notebook:
- Loads the trained model checkpoint
- Runs inference on full test set (644 samples)
- Generates predictions for all test samples
- Saves results locally + backs up to Drive
- Shows sample predictions vs ground truth

**Expected output:**
```
✓ Checkpoint copied from Google Drive
✓ Inference complete
  Predictions shape: [644, 512, 512]
  Targets shape: [644, 512, 512]
✓ Saved predictions to inference_results/predictions.pt
✓ Backed up to Google Drive: /content/drive/MyDrive/RETINNA_checkpoints/predictions.pt

[Sample prediction visualizations]
```

### 5. Run Notebook 05 (Analysis & Metrics) — CPU Recommended

**Machine type:** CPU only (metrics are CPU-based sklearn operations)  
**Estimated time:** 5-10 minutes  
**GPU hours saved:** ~10 min (use for training instead)  
**Session type:** New session on CPU, or same session as 04

Open `notebooks/05_analysis.ipynb`:
1. **Runtime → Change runtime type → CPU** (frees GPU for other use)
2. Run all cells
3. Automatically copies predictions.pt from Drive if needed

This notebook loads predictions from 04 and computes:
- **Accuracy, Precision, Recall, F1-score, IoU**
- **Confusion matrix** (TP/FP/FN/TN)
- **ROC curve with AUC**
- **Threshold analysis** (metrics vs probability threshold)
- **Error analysis** — samples of false positives/negatives
- Saves metrics.json + backs up all results to Drive

**Expected output:**
```
=== Test Set Metrics (Threshold 0.5) ===
Accuracy:     0.9234
Precision:    0.7123
Recall:       0.6845
F1-Score:     0.6982
IoU:          0.5201
ROC AUC:      0.9156
Specificity:  0.9876

✓ Saved confusion_matrix.png
✓ Saved roc_curve.png
✓ Saved threshold_analysis.png
✓ Saved error_analysis.png
✓ Saved metrics.json
✓ All results backed up to Google Drive
```

## Tips & Troubleshooting

### Switching Machine Types in Colab

**To change from CPU to GPU:**
1. **Runtime → Change runtime type**
2. Select "GPU" from the dropdown
3. Click "Save"
4. Restart kernel (runtime will auto-restart)

**To change from GPU to CPU:**
1. **Runtime → Change runtime type**
2. Select "CPU" from the dropdown
3. Click "Save"
4. Restart kernel

**Free tier GPU limits:**
- ~4-8 hours per day (varies)
- Up to 8 hours per session
- Idle timeout: 30 minutes with no code executing

### If Running Out of GPU Memory

In notebook 03, reduce batch size:
```python
dataloaders = get_dataloaders(
    batch_size=2,  # Reduce from 4
    ...
)
```

Or reduce image size in dataset loading (requires modifying dataset.py).

### If Training Disconnects

The Colab kernel continues running even if your browser disconnects. You can:
1. Refresh browser and reconnect
2. Check if checkpoint exists on Drive: `/content/drive/MyDrive/RETINNA_checkpoints/best_model.pth`
3. 04_inference will auto-load from Drive checkpoint

### GPU Not Available

If you get "CPU" device message:
- Free tier may run CPU-only during peak hours
- Try again later, or run in early morning (UTC)
- Or run with fewer epochs on CPU (slower, ~10x)
- CPU training is viable for small experiments (5-10 epochs)

## Recommended Workflow (Full Baseline)

Complete workflow with optimal resource usage:

| Step | Notebook | Machine | Time | GPU Hours |
|------|----------|---------|------|-----------|
| 1 | 01_data_pipeline | CPU | 10 min | — |
| 2 | 02_exploratory_analysis | CPU | 10 min | — |
| 3 | 03_training | **GPU (T4)** | 20 min | 0.33 |
| 4 | 04_inference | **GPU (T4)** | 10 min | 0.17 |
| 5 | 05_analysis | CPU | 5 min | — |

**Total GPU time:** ~0.5 hours (well within free tier limits)  
**Total wall time:** ~55 minutes (can run notebooks sequentially)  
**Cost:** Free (on Colab free tier)

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
