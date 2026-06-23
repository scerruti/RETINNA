# Project Structure

Complete overview of RETINNA repository organization and file purposes.

## Directory Layout

```
RETINNA/
├── docs/                                    # Documentation
│   ├── QUICK_START.md                       # How to run on Colab (start here!)
│   ├── PROJECT_STRUCTURE.md                 # This file
│   ├── TRAINING_PROCESS.md                  # Training loop, loss, hyperparameters
│   ├── DEVICE_HANDLING.md                   # CPU/GPU compatibility patterns
│   ├── ARCHITECTURE_RATIONALE.md            # Why U-Net, not FCN; PA3 comparison
│   ├── DATA_CLEANING_USAGE.md               # Data validation and split strategy
│   └── INFERENCE_GUIDE.md                   # Using inference for new imagery
│
├── notebooks/                               # Jupyter notebooks (Colab)
│   ├── 01_data_pipeline.ipynb               # Load dataset, validate, setup
│   ├── 02_exploratory_analysis.ipynb        # EDA: spectral stats, NDVI, anomalies
│   ├── 03_training.ipynb                    # Train model with interactive slider
│   ├── 04_inference.ipynb                   # Generate predictions on test set
│   └── 05_analysis.ipynb                    # Metrics, confusion matrix, visualizations
│
├── src/                                     # Python modules
│   ├── __init__.py
│   │
│   ├── unet.py                              # U-Net model (31M parameters)
│   │   ├── DoubleConv: Conv→BN→ReLU×2
│   │   ├── EncoderBlock: DoubleConv + MaxPool
│   │   ├── DecoderBlock: TransposeConv + Skip + DoubleConv
│   │   └── UNet: Full encoder-decoder with skip connections
│   │
│   ├── dataset.py                           # Data loading pipeline
│   │   ├── CaBuArDataset: Wrapper for TorchGeo CaBuAr
│   │   │   - Bi-temporal [B, 2, 12, 512, 512]
│   │   │   - Optional normalization (÷10000)
│   │   │   - Ground truth masks (binary)
│   │   └── get_dataloaders(): Creates train/val/test DataLoaders
│   │
│   ├── device_utils.py                      # Device management
│   │   ├── get_device(): Returns GPU or CPU
│   │   └── move_to_device(): Recursively moves tensors/dicts/lists
│   │
│   ├── colab_utils.py                       # Google Colab utilities
│   │   ├── setup_colab_environment(): Mount Drive, verify GPU
│   │   ├── get_dataset_cache_path(): Create cache directory
│   │   └── setup_cabuaur_cached(): Setup dataset with Drive cache
│   │
│   └── data_cleaning.py                     # Data validation
│       ├── validate_split(): Check for corruption, NaN, shape mismatches
│       └── validate_dataset(): Validate all splits
│
├── train.py                                 # Training script (CLI)
│   ├── BCEDiceLoss: Hybrid loss for class imbalance
│   ├── compute_iou(): Intersection over Union metric
│   ├── train_epoch(): Single epoch loop
│   ├── validate(): Validation with metrics
│   ├── save_checkpoint() / load_checkpoint(): Model persistence
│   └── main(): Full training orchestration
│   └── Argument parsing: --epochs, --batch-size, --learning-rate, etc.
│
├── README.md                                # Project overview
├── .gitignore                               # Git ignore patterns
├── requirements.txt                         # Python dependencies
└── LICENSE                                  # MIT License

# Generated During Runs

checkpoints/                                 # CLI training checkpoints
├── best.pth                                 # Best model
├── latest.pth                               # Latest model
└── config.json                              # Training config

checkpoints_notebook/                        # Notebook training checkpoints
├── best_model.pth                           # Best model from 03
└── config.json                              # Training config

inference_results/                           # Inference output
└── predictions.pt                           # Predictions from 04

/content/drive/MyDrive/RETINNA_DATA/         # Google Drive cache (Colab)
└── cabuaur/512x512.hdf5                     # CaBuAr dataset (~6GB)
```

## File Purposes

### Core Modules (`src/`)

#### `unet.py` (Model)
- **Purpose**: U-Net semantic segmentation architecture
- **Key classes**:
  - `DoubleConv`: Double convolution blocks
  - `EncoderBlock`: Downsampling path
  - `DecoderBlock`: Upsampling path with skip connections
  - `UNet`: Full model (in_channels=24, out_channels=2)
- **Key methods**:
  - `forward()`: Inference
  - `get_parameter_count()`: Model size

#### `dataset.py` (Data Loading)
- **Purpose**: CaBuAr dataset access and DataLoaders
- **Key classes**:
  - `CaBuArDataset`: TorchGeo wrapper with normalization
- **Key functions**:
  - `get_dataloaders()`: Create train/val/test loaders
- **Output shapes**: 
  - Images: [B, 2, 12, 512, 512] → flattened to [B, 24, 512, 512]
  - Masks: [B, 1, 512, 512]

#### `device_utils.py` (Device Management)
- **Purpose**: CPU/GPU abstraction following PA3 pattern
- **Key functions**:
  - `get_device()`: Detect GPU, fallback to CPU
  - `move_to_device()`: Recursively move data structures to device
- **Usage**: Enables code to run on any machine (local CPU, Colab GPU, etc.)

#### `colab_utils.py` (Colab Integration)
- **Purpose**: Google Drive mounting and dataset caching
- **Key functions**:
  - `setup_colab_environment()`: Mount Drive, verify GPU
  - `setup_cabuaur_cached()`: Cache dataset on Drive (~6GB download once)
- **Benefit**: Avoid re-downloading dataset every Colab session

#### `data_cleaning.py` (Data Validation)
- **Purpose**: Detect corrupted/empty tiles
- **Key functions**:
  - `validate_split()`: Check for NaN, zero bands, shape mismatches
  - `validate_dataset()`: Validate all splits
- **Output**: Report with corrupted tile counts

### Training (`train.py`)

**Purpose**: Standalone training script (alternative to notebook)

**Key components**:
- `BCEDiceLoss`: Binary CE + Dice for class imbalance
- `compute_iou()`: IoU metric for burned class
- `train_epoch()`: Single epoch loop (forward → backward → step)
- `validate()`: Validation with loss + IoU
- `save_checkpoint()`: Save best + latest models
- `main()`: Full training orchestration

**Usage**:
```bash
python train.py --epochs 50 --batch-size 32 --learning-rate 0.0005
```

**Output**: Checkpoints saved to `checkpoints/` directory

### Notebooks (`notebooks/`)

All notebooks follow this pattern:
1. **Clone repo** (if running on Colab)
2. **Install dependencies** (pip install)
3. **Setup** (imports, device, paths)
4. **Load data** (with caching)
5. **Execute workflow** (training, inference, analysis)
6. **Visualize results** (plots, metrics)

#### `01_data_pipeline.ipynb`
- **Purpose**: Single entry point for data setup
- **Consolidates**: Dataset loading, validation, DataLoader creation
- **Output**: Verified dataloaders ready for training

#### `02_exploratory_analysis.ipynb`
- **Purpose**: Understand dataset characteristics
- **Contents**: Band statistics, NDVI, spectral signatures, anomalies
- **Optional**: Can skip if only training

#### `03_training.ipynb`
- **Purpose**: Model training with interactive controls
- **Special**: Epochs slider for quick experimentation
- **Output**: `checkpoints_notebook/best_model.pth`

#### `04_inference.ipynb`
- **Purpose**: Generate predictions on test set
- **Input**: Best checkpoint from 03
- **Output**: `inference_results/predictions.pt` (for 05)

#### `05_analysis.ipynb`
- **Purpose**: Detailed evaluation and metrics
- **Contents**: IoU, precision, recall, F1, confusion matrix, visualizations
- **Output**: Baseline evaluation report

### Documentation (`docs/`)

| File | Purpose |
|------|---------|
| QUICK_START.md | Step-by-step Colab workflow (start here!) |
| PROJECT_STRUCTURE.md | This file — directory overview |
| TRAINING_PROCESS.md | Training loop, loss functions, hyperparameters |
| DEVICE_HANDLING.md | CPU/GPU compatibility, PA3 pattern |
| ARCHITECTURE_RATIONALE.md | Why U-Net; comparison to FCN (PA3) |
| DATA_CLEANING_USAGE.md | Data validation and split strategy |
| INFERENCE_GUIDE.md | Using inference for new imagery |

## Data Flow

```
CaBuAr Dataset (TorchGeo)
        ↓
    [01: Load & Validate]
        ↓
    [02: Explore] (optional)
        ↓
[Train / Val / Test Splits]
        ↓
[03: Train Model] → checkpoint
        ↓
[04: Inference] → predictions.pt
        ↓
[05: Analysis] → metrics & report
```

## Key Statistics

| Item | Value |
|------|-------|
| **Model** | U-Net (custom PyTorch) |
| **Parameters** | 31.1 million |
| **Input shape** | [B, 24, 512, 512] |
| **Output shape** | [B, 2, 512, 512] |
| **Training samples** | 3,098 |
| **Validation samples** | 644 |
| **Test samples** | 644 |
| **Dataset size** | ~6 GB (HDF5) |
| **Training time** | 50 epochs ≈ 50 min (T4 GPU) |

## Development Notes

### Module Dependencies

```
train.py
├── src.unet (model)
├── src.dataset (dataloaders)
├── src.device_utils (GPU/CPU)
└── src.colab_utils (Drive caching, optional)

notebooks/03_training.ipynb
├── src.unet
├── src.dataset
├── src.device_utils
├── src.colab_utils
└── ipywidgets (slider)

notebooks/04_inference.ipynb
├── src.unet
├── src.dataset
└── src.device_utils
```

### Testing Checklist

- [ ] Clone repo works on Colab
- [ ] All pip installs succeed
- [ ] 01_data_pipeline loads dataset (no corruption)
- [ ] 03_training runs 1 epoch without errors
- [ ] 03_training saves checkpoint
- [ ] 04_inference loads checkpoint and runs
- [ ] 05_analysis computes metrics

## Common Workflows

### Quick Test (5 min)
1. Run 01_data_pipeline
2. Run 03_training with 1 epoch
3. Run 04_inference

### Full Baseline (45 min)
1. Run 01-02
2. Run 03 with 20 epochs
3. Run 04-05

### Hyperparameter Tuning (2-4 hours)
1. Run 01
2. Run 03 multiple times with different settings (epochs, learning_rate)
3. Compare checkpoints with 04-05

### Production Inference
1. Load best checkpoint from training
2. Use 04_inference pattern for any new imagery
3. Visualize with 05_analysis patterns

---

See [QUICK_START.md](QUICK_START.md) to get running on Colab.
