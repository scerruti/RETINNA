# Baseline Results: U-Net Burn Detection Model

**Completed**: Day 3 Sprint  
**Issue**: #11 (Training Script)  
**Model**: U-Net with 31.1M parameters  
**Dataset**: CaBuAr (California Burned Areas)  

## Training Summary

| Metric | Value |
|--------|-------|
| **Epochs** | 20 |
| **Device** | T4 GPU (Colab Pro) |
| **Training Time** | ~20 minutes |
| **Best Validation IoU** | 0.47 |
| **Final Train Loss** | 0.24 |
| **Final Val Loss** | 0.31 |
| **Batch Size** | 4 |
| **Learning Rate** | 0.0005 |
| **Optimizer** | Adam |
| **Loss Function** | BCE + Dice (0.5 weight each) |

## Training Curves

![Training Curves](training_curves.png)

### Loss Curve (Left)
- **Train Loss** (blue): Decreases smoothly from 0.56 → 0.24
  - Indicates model is learning consistently
  - No divergence or NaN values
  
- **Val Loss** (orange): Decreases from 0.50 → 0.31
  - More volatile than train loss (expected for validation set)
  - General downward trend indicates generalization
  - Some spikes (epochs 5, 17) suggest occasional difficult batches

### IoU Curve (Right)
- **Validation IoU** (green): Increases from 0.38 → 0.47
  - Starting point (0.38) is reasonable for random initialization
  - Best performance: **0.47 at epoch 13**
  - Validation IoU metric for burned class only (binary segmentation)
  - Noisy trajectory is normal (class is sparse, ~10% of pixels)

## Model Performance

### What These Metrics Mean

**IoU (Intersection over Union) = 0.47:**
- For every pixel predicted as burned, ~47% overlap with ground truth on average
- Room for improvement but solid baseline
- Typical ranges:
  - 0.30-0.40: Basic detection working
  - 0.40-0.60: Good baseline (we are here)
  - 0.60-0.75: Well-tuned model
  - 0.75+: Excellent performance

**Loss Values:**
- BCE+Dice loss combines two complementary objectives
- Lower is better; 0.31 final validation loss is reasonable
- Hybrid loss helps with class imbalance (90% unburned, 10% burned)

### Training Dynamics

✅ **Healthy Signs:**
- Monotonic decrease in training loss (no oscillation)
- Validation loss trending downward overall
- IoU improving from epoch 1-13 (best checkpoint)
- No NaN or infinity values
- Model converging toward optimal weights

⚠️ **Observations:**
- Validation loss spikes in later epochs (5, 17)
  - Possible: difficult validation batches
  - Solution: Could use validation loss monitoring for early stopping
- IoU plateaus after epoch 13
  - Possible: model near local optimum with current hyperparameters
  - Solution: Hyperparameter tuning (learning rate, epochs, regularization)

## Checkpoint Information

**Best Model**: `checkpoints_notebook/best_model.pth`
- Saved at epoch 13
- Validation IoU: 0.47
- Model state dict: 31.1M parameters
- Ready for inference on test set

## Next Steps

### Immediate (Issue #12: Baseline Evaluation)
1. **Run inference** (04_inference.ipynb) on full test set
2. **Compute detailed metrics**:
   - Precision, Recall, F1-score
   - Confusion matrix
   - Per-class statistics
   - Error analysis
3. **Generate visualizations** of predictions vs ground truth

### Short Term (Hyperparameter Tuning)
- **Learning rate search**: Try 0.001, 0.0001
- **Batch size**: Experiment with 8, 16 to affect gradient stability
- **Loss weights**: Adjust BCE/Dice ratio for class balance
- **Regularization**: Add dropout or L2 to prevent overfitting

### Medium Term (Feature Engineering)
- **Data augmentation**: Rotation, flipping, elastic deformation
- **Multi-scale training**: Pyramid approach for different tile sizes
- **Ensemble methods**: Train multiple models with different seeds

### Long Term (Advanced)
- **Multi-class burn severity**: Extend from binary to severity levels
- **Cross-satellite transfer**: Test on Landsat 8 imagery
- **Production deployment**: Export model, create inference API

## Comparison to PA3

| Aspect | PA3 (Scene Segmentation) | RETINNA (Burn Detection) |
|--------|--------------------------|--------------------------|
| **Task** | 27-class scene labels | Binary burn classification |
| **Input** | 3-channel RGB | 24-channel bi-temporal |
| **Model** | FCN (fully convolutional) | U-Net (encoder-decoder) |
| **Key Metric** | Accuracy per class | IoU for burned class |
| **Class Balance** | Balanced classes | Imbalanced (90% unburned) |
| **Baseline IoU** | ~0.50 | **0.47** (comparable!) |

**Insight**: Similar IoU performance despite different domains (scene vs. burn detection) validates our U-Net architecture choice and loss function design.

## Files Generated

- `checkpoints_notebook/best_model.pth` — Trained model weights
- `checkpoints_notebook/config.json` — Training configuration
- `docs/baseline_training_curves.png` — This document's loss/IoU curves
- `docs/BASELINE_RESULTS.md` — This document

## Reproducibility

To reproduce this baseline:

```bash
# On Colab
1. Run notebooks/01_data_pipeline.ipynb
2. Run notebooks/03_training.ipynb with epochs_slider = 20
3. Save output as docs/training_curves.png
```

**Configuration preserved in**: `checkpoints_notebook/config.json`

---

**Author**: Stephen Cerruti  
**Date**: 2026-06-23  
**Dataset**: CaBuAr (TorchGeo)  
**Status**: ✅ Complete, ready for inference
