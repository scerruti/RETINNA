import pytest
import tempfile
import numpy as np
from pathlib import Path
import h5py
import json

from src.data_loader import CaBuArDataLoader


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_hdf5_file(temp_cache_dir):
    """Create a minimal test HDF5 file matching CaBuAr structure."""
    filepath = Path(temp_cache_dir) / "test_california_1.hdf5"

    with h5py.File(filepath, 'w') as f:
        fold = f.create_group('0')
        wildfire = fold.create_group('wildfire_001')

        # Create sample data
        pre_fire_data = np.random.rand(11, 32, 32).astype(np.float32)
        post_fire_data = np.random.rand(11, 32, 32).astype(np.float32)
        mask_data = np.random.randint(0, 2, (32, 32)).astype(np.uint8)

        wildfire.create_dataset('pre_fire', data=pre_fire_data)
        wildfire.create_dataset('post_fire', data=post_fire_data)
        wildfire.create_dataset('mask', data=mask_data)

    return filepath


def test_loader_initialization(temp_cache_dir):
    """Test that loader initializes and creates cache directory."""
    loader = CaBuArDataLoader(cache_dir=temp_cache_dir)
    assert loader.cache_dir == Path(temp_cache_dir)
    assert loader.dataset is None
    assert loader.stats == {}


def test_parse_hdf5_file(sample_hdf5_file):
    """Test parsing a single HDF5 file."""
    loader = CaBuArDataLoader()

    with h5py.File(sample_hdf5_file, 'r') as h5_file:
        samples = loader._parse_hdf5_file(h5_file)

    assert len(samples) == 1
    assert 'wildfire_id' in samples[0]
    assert 'pre_fire' in samples[0]
    assert 'post_fire' in samples[0]
    assert 'mask' in samples[0]
    assert samples[0]['pre_fire'].shape == (11, 32, 32)
    assert samples[0]['mask'].shape == (32, 32)


def test_parse_hdf5_file_missing_fold(temp_cache_dir):
    """Test error handling for missing fold in HDF5 file."""
    filepath = Path(temp_cache_dir) / "bad_file.hdf5"
    with h5py.File(filepath, 'w') as f:
        f.create_group('1')  # Wrong fold key

    loader = CaBuArDataLoader()
    with h5py.File(filepath, 'r') as h5_file:
        with pytest.raises(ValueError, match="Expected fold '0' not found"):
            loader._parse_hdf5_file(h5_file)


def test_compute_stats_no_dataset():
    """Test that compute_stats raises error if dataset not loaded."""
    loader = CaBuArDataLoader()
    with pytest.raises(ValueError, match="Dataset not loaded"):
        loader.compute_stats()


def test_compute_stats_with_data(sample_hdf5_file):
    """Test computing statistics with sample data."""
    loader = CaBuArDataLoader()

    # Manually load test data
    with h5py.File(sample_hdf5_file, 'r') as h5_file:
        loader.dataset = loader._parse_hdf5_file(h5_file)

    stats = loader.compute_stats()

    assert stats['total_samples'] == 1
    assert 'total_pixels' in stats
    assert 'burned_pixels' in stats
    assert 'unburned_pixels' in stats
    assert 'burned_percent' in stats
    assert 'unburned_percent' in stats
    assert stats['total_pixels'] == 32 * 32


def test_compute_stats_zero_pixels(temp_cache_dir):
    """Test stats computation with empty masks (divide by zero edge case)."""
    loader = CaBuArDataLoader()
    loader.dataset = [{
        'wildfire_id': 'test',
        'pre_fire': np.zeros((3, 32, 32)),
        'post_fire': np.zeros((3, 32, 32)),
        'mask': np.zeros((32, 32))  # All unburned
    }]

    stats = loader.compute_stats()

    # All pixels are unburned (mask == 0)
    assert stats['burned_pixels'] == 0
    assert stats['unburned_pixels'] == 32 * 32
    assert stats['burned_percent'] == 0.0
    assert stats['unburned_percent'] == 100.0


def test_save_stats(sample_hdf5_file, temp_cache_dir):
    """Test saving statistics to JSON."""
    loader = CaBuArDataLoader()

    # Load and compute stats
    with h5py.File(sample_hdf5_file, 'r') as h5_file:
        loader.dataset = loader._parse_hdf5_file(h5_file)
    loader.compute_stats()

    # Save stats
    stats_file = Path(temp_cache_dir) / "test_stats.json"
    loader.save_stats(filepath=str(stats_file))

    # Verify file was created and contains valid JSON
    assert stats_file.exists()
    with open(stats_file) as f:
        saved_stats = json.load(f)

    assert saved_stats['total_samples'] == loader.stats['total_samples']
    assert saved_stats['burned_percent'] == loader.stats['burned_percent']


def test_visualize_samples_no_dataset(temp_cache_dir):
    """Test that visualize_samples raises error if dataset not loaded."""
    loader = CaBuArDataLoader()
    viz_dir = Path(temp_cache_dir) / "viz"

    with pytest.raises(ValueError, match="Dataset not loaded"):
        loader.visualize_samples(save_dir=str(viz_dir))


def test_visualize_samples(sample_hdf5_file, temp_cache_dir):
    """Test visualization creation."""
    loader = CaBuArDataLoader()

    # Load test data
    with h5py.File(sample_hdf5_file, 'r') as h5_file:
        loader.dataset = loader._parse_hdf5_file(h5_file)

    # Visualize
    viz_dir = Path(temp_cache_dir) / "viz"
    loader.visualize_samples(num_samples=1, save_dir=str(viz_dir))

    # Check that visualization was created
    output_file = viz_dir / "sample_000.png"
    assert output_file.exists()
    assert output_file.stat().st_size > 0
