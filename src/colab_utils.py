"""
Google Colab utilities for RETINNA project.

Handles Google Drive mounting, dataset caching, and path management.
"""

import os
from pathlib import Path


def setup_colab_environment():
    """
    Setup Colab environment: mount Google Drive and verify GPU availability.

    Returns:
        dict: Configuration with paths and device info
    """
    try:
        from google.colab import drive
        import torch

        # Mount Google Drive
        drive.mount('/content/drive', force_remount=False)
        drive_ready = True
        print("✓ Google Drive mounted at /content/drive")
    except ImportError:
        drive_ready = False
        print("⚠ Not running on Google Colab (Google Drive unavailable)")

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name}")
            device = "cuda"
        else:
            print("⚠ GPU not available, using CPU")
            device = "cpu"
    except ImportError:
        device = "cpu"

    return {
        'drive_ready': drive_ready,
        'device': device,
        'drive_path': '/content/drive/MyDrive' if drive_ready else None
    }


def get_dataset_cache_path(dataset_name='cabuaur', env_config=None):
    """
    Get or create cached dataset path on Google Drive.

    Args:
        dataset_name (str): Name of dataset (e.g., 'cabuaur', 'landsat8')
        env_config (dict): Environment config from setup_colab_environment()

    Returns:
        str: Path to cache directory
    """
    if env_config is None:
        env_config = setup_colab_environment()

    if not env_config['drive_ready']:
        print("⚠ Google Drive not available, using /tmp/ (will not persist across runtimes)")
        cache_path = f'/tmp/{dataset_name}'
    else:
        # Create RETINNA data cache folder on Drive
        drive_path = env_config['drive_path']
        cache_path = os.path.join(drive_path, 'RETINNA_DATA', dataset_name)

        # Ensure directory exists
        os.makedirs(cache_path, exist_ok=True)
        print(f"✓ Dataset cache path: {cache_path}")

    os.makedirs(cache_path, exist_ok=True)
    return cache_path


def setup_cabuaur_cached(env_config=None):
    """
    Setup CaBuAr dataset with Google Drive caching.

    This ensures the dataset is downloaded once to Google Drive,
    then reused across notebooks.

    Args:
        env_config (dict): Environment config from setup_colab_environment()

    Returns:
        str: Path to cached CaBuAr dataset

    Example:
        >>> env = setup_colab_environment()
        >>> cabuaur_path = setup_cabuaur_cached(env)
        >>> from torchgeo.datasets import CaBuAr
        >>> dataset = CaBuAr(root=cabuaur_path, download=True, split='test')
    """
    cache_path = get_dataset_cache_path('cabuaur', env_config)
    print(f"✓ CaBuAr will use cache: {cache_path}")
    return cache_path


def setup_imports():
    """
    Install and import common dependencies for RETINNA notebooks.

    Run this once at the start of your notebook.
    """
    print("Installing dependencies...")

    import subprocess
    import sys

    packages = [
        'torch',
        'torchvision',
        'torchgeo',
        'numpy',
        'matplotlib',
        'scipy',
        'scikit-learn',
        'pandas',
        'rasterio',
        'shapely',
    ]

    for package in packages:
        try:
            __import__(package)
            print(f"  ✓ {package} already installed")
        except ImportError:
            print(f"  Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

    print("✓ All dependencies ready")
