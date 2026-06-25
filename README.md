# RETINNA: California Wildfire Semantic Segmentation

Semantic segmentation of wildfire burn scars in California satellite imagery using a custom PyTorch U-Net implementation.

## Project Objective

RETINNA builds and trains a Convolutional Neural Network (CNN) from scratch to perform **pixel-level semantic segmentation** on satellite imagery. The model evaluates pre- and post-fire landscape images to identify and map the precise boundaries of wildfire burn scars.

Unlike binary classification ("Is there a fire?"), this semantic segmentation approach outputs a spatial mask, enabling:
- Direct calculation of damaged acreage
- Precise visualization of fire perimeters
- Fine-grained boundary analysis for landscape recovery planning

## Dataset

This project leverages the **CaBuAr (California Burned Areas)** dataset from [Hugging Face](https://huggingface.co/datasets/DarthReca/california_burned_areas), loaded via the **TorchGeo** library on Colab.

**✅ Dataset Access**: Fully working on Google Colab via TorchGeo. Automatically cached to Google Drive on first run.

### Input Data
- **Bi-temporal satellite imagery**: Pre-fire and post-fire passes captured by Sentinel-2
- **Resolution**: 512×512 pixel tiles with 10m ground sampling distance
- **Spectral bands**: Multi-spectral imagery including visible and near-infrared wavelengths

### Ground Truth
- **Official burn perimeters**: Provided by CAL FIRE
- **Format**: Rasterized binary masks (1 = Burned, 0 = Unburned)

### Advantage
Using a curated dataset eliminates heavy data engineering required to align raw satellite geospatial data (GeoTIFFs) with vector shapefiles, allowing focus on model architecture and deep learning mechanics.

## Technical Stack & Environment

### Local Development
- **IDE**: Visual Studio Code with Claude Code
- **Purpose**: Rapid, iterative writing of modular Python scripts and utility functions

### Cloud Compute
- **Platform**: Google Colab Pro (or equivalent)
- **Purpose**: GPU-accelerated PyTorch model training and tensor operations

### Core Libraries
```
torch>=2.0.0
torchvision>=0.15.0
datasets>=2.0.0
matplotlib>=3.5.0
numpy>=1.20.0
```

## Model Architecture: U-Net

The project implements a custom U-Net architecture in PyTorch—the industry standard for precise image segmentation.

### Encoder (Contracting Path)
- Convolutional and pooling layers downsample the image
- Captures context and deep feature representations
- Learns spectral signatures (e.g., near-infrared reflectance drop in charred vegetation)

### Decoder (Expanding Path)
- Transposed convolutions upsample feature maps back to original resolution
- Reconstructs spatial information progressively

### Skip Connections
- High-resolution spatial information passes directly from encoder to decoder
- Prevents loss of fine boundary details during upsampling
- Crucial for accurate burn scar delineation

## Training & Evaluation Strategy

### Loss Function
Hybrid loss combining:
- **Binary Cross-Entropy (BCE)**: Standard classification loss
- **Dice Loss**: Mitigates class imbalance (burn scars occupy <10% of typical tiles)

The combined loss ensures the model is heavily penalized for simply predicting the majority "unburned" class.

### Primary Metric
- **Intersection over Union (IoU)**: Jaccard Index calculated specifically for the positive "burned" class
- Evaluates boundary accuracy and spatial overlap with ground truth

### Optimization
- **Optimizer**: Adam
- **Tracking**: Training and validation loss curves
- **Prevention**: Early stopping and validation monitoring to prevent overfitting

## Expected Deliverables

By the end of the development sprint:

1. **Custom PyTorch Dataset Pipeline**
   - Efficient loading and preprocessing of bi-temporal satellite imagery
   - Data augmentation strategies for improved generalization

2. **Modular U-Net Class**
   - Scratch-built PyTorch implementation
   - Configurable encoder/decoder depths and channel dimensions
   - Clean, reusable architecture for future projects

3. **Trained Model Weights**
   - `.pth` checkpoint file
   - Training history and hyperparameter documentation

4. **Inference Script**
   - Ingests unseen pre/post-fire landscape image pairs
   - Outputs predicted burn perimeter masks
   - Batch processing capability

5. **Data Visualizations**
   - Raw post-fire satellite image
   - CAL FIRE ground truth mask
   - AI-generated prediction mask
   - Side-by-side comparison plots
   - Quantitative metrics overlay

## Project Timeline

5-day sprint structure with daily sprint issues tracking progress:
- **Day 1**: Dataset exploration, data pipeline, initial EDA
- **Day 2**: U-Net architecture implementation, training setup
- **Day 3**: Model training, validation metrics tracking
- **Day 4**: Fine-tuning, hyperparameter optimization
- **Day 5**: Inference pipeline, visualizations, documentation

## Getting Started

See **[QUICK_START.md](docs/QUICK_START.md)** for step-by-step instructions to run on Google Colab.

### Quick Overview
1. **Data Pipeline**: `notebooks/01_data_pipeline.ipynb` — Load and validate dataset
2. **Exploration**: `notebooks/02_exploratory_analysis.ipynb` — Analyze spectral properties
3. **Training**: `notebooks/03_training.ipynb` — Train model with interactive controls
4. **Inference**: `notebooks/04_inference.ipynb` — Generate predictions on test set
5. **Evaluation**: `notebooks/05_analysis.ipynb` — Compute metrics and visualizations

### Training Script (Advanced)
```bash
python train.py --epochs 50 --batch-size 32 --learning-rate 0.0005
```

See [TRAINING_PROCESS.md](docs/TRAINING_PROCESS.md) for detailed configuration options.

## Project Status

**Phase II: Spectral Relabeling & U-Net Training (In Progress)**

### Completed ✅
- **Phase II_01**: Spectral relabeling using RdNBR (7-class severity labels generated)
- **Phase II_03**: Metadata extraction (298 fires, 534 samples catalogued)
- **Phase II_02 Implementation**: 8-channel U-Net with z-score normalization + augmentation (awaiting Colab GPU execution)

### In Progress ⏳
- Phase II_02 Colab Training: Run II_01 and II_02 notebooks on Colab GPU
- Test set evaluation and comparison to baseline

### Architecture Evolution
- **Initial approach (PA3)**: 4-channel difference model (Post - Pre)
- **Current approach (Phase II_02)**: 8-channel separate images (Pre + Post concatenated) with z-score normalization
  - Enables flexible change detection learning
  - Better generalization to NAIP imagery in Phase III
  - Data augmentation (flip, rotate, zoom/crop)

### Next Steps
1. Execute Phase II_02 on Colab (5-10 min II_01 + 20-30 min II_02 training)
2. Evaluate test set performance
3. Prepare Phase III NAIP inference pipeline

**Documentation**: See [docs/PHASE_II_INDEX.md](docs/PHASE_II_INDEX.md) for complete Phase II overview.  
**Execution Guide**: [docs/PHASE_II_02_COLAB_EXECUTION_8CH.md](docs/PHASE_II_02_COLAB_EXECUTION_8CH.md)

## References

- U-Net: Convolutional Networks for Biomedical Image Segmentation ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597))
- CaBuAr Dataset: [Hugging Face - california_burned_areas](https://huggingface.co/datasets/DarthReca/california_burned_areas)
- Sentinel-2 Satellite Imagery: [ESA Copernicus Program](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)

## Program Context

This project is developed as part of **RETINNA** (Research Experience for Teachers in Interdisciplinary Artificial Intelligence), a 6-week summer program at UC San Diego for K-12 and community college computer science teachers to develop expertise in deep learning and AI curriculum design.

**RETINNA Program**: https://sites.google.com/ucsd.edu/ucsdretinna/

The wildfire burn scar segmentation work also contributes to a broader grant initiative focused on wildfire risk assessment and landscape recovery optimization using satellite imagery and machine learning.

---

**Author**: Stephen Cerruti  (steve.cerruti@gmail.com)
**Advisor**: Garrison Cottrell, UC San Diego (gary@ucsd.edu)  
**TA**: Zhining Chen (zhc008@ucsd.edu)  
**Acknowledgments**: Claude Code, GitHub Copilot, Google Gemini  
**License**: MIT
