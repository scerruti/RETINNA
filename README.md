# RETINNA: California Wildfire Semantic Segmentation

Semantic segmentation of wildfire burn scars in California satellite imagery using a custom PyTorch U-Net implementation.

## Project Objective

RETINNA builds and trains a Convolutional Neural Network (CNN) from scratch to perform **pixel-level semantic segmentation** on satellite imagery. The model evaluates pre- and post-fire landscape images to identify and map the precise boundaries of wildfire burn scars.

Unlike binary classification ("Is there a fire?"), this semantic segmentation approach outputs a spatial mask, enabling:
- Direct calculation of damaged acreage
- Precise visualization of fire perimeters
- Fine-grained boundary analysis for landscape recovery planning

## Dataset

This project leverages the **CaBuAr (California Burned Areas)** dataset, accessible via [Hugging Face](https://huggingface.co/datasets/DarthReca/california_burned_areas).

**Note**: See [Dataset Access Documentation](docs/CABUАР_DATASET_ACCESS.md) for details on dataset loading, HDF5 plugin requirements, and troubleshooting.

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

### Prerequisites
```bash
pip install -r requirements.txt
```

### Training
```python
python train.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

### Inference
```python
python inference.py --model_weights model.pth --image_pair pre_fire.tif post_fire.tif
```

## Project Status

Currently in active development. Track progress using daily sprint issues in the repository.

## References

- U-Net: Convolutional Networks for Biomedical Image Segmentation ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597))
- CaBuAr Dataset: [Hugging Face - california_burned_areas](https://huggingface.co/datasets/DarthReca/california_burned_areas)
- Sentinel-2 Satellite Imagery: [ESA Copernicus Program](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)

## Grant Project

This project is part of a broader grant initiative focused on wildfire risk assessment and landscape recovery optimization using satellite imagery and machine learning.

---

**Author**: Stephen Cerruti  
**Advisor**: Gary Cottrell, UC San Diego  
**Acknowledgments**: Claude Code, GitHub Copilot, Google Gemini  
**License**: MIT
