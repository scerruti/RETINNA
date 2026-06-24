# Colab Google Drive Setup for Phase 3

**Purpose**: Robust handling of Google Drive caching with automatic reconnection  
**Location**: `src/drive_utils.py`  
**For**: Phase 3+ notebooks running on Colab

---

## Quick Start

Add this to the top of any Phase 3+ Colab notebook:

```python
# Setup and connect to Google Drive
import sys
sys.path.insert(0, '/content/RETINNA/src')
from drive_utils import setup_drive_for_colab

# Mount drive with automatic testing
drive_manager = setup_drive_for_colab(verbose=True)

if drive_manager is None:
    print("⚠️  WARNING: Could not connect to Google Drive")
    print("   Continuing with local-only mode...")
else:
    print("✓ Google Drive ready for caching")
```

---

## DriveManager API

### Mounting

```python
from drive_utils import DriveManager

manager = DriveManager(mount_point="/content/drive")

# Mount
manager.mount(verbose=True)

# Test access
manager.test_access(verbose=True)
```

### Reconnection (if Drive disconnects)

```python
# Automatic reconnect with retry
success = manager.reconnect(max_attempts=3, verbose=True)

if not success:
    print("Could not reconnect. Check Drive status.")
```

### Saving Data with Timestamps

```python
import torch
import numpy as np

# Save tensor
labels_tensor = torch.ones(100, 512, 512)
path = manager.save_with_timestamp(
    labels_tensor,
    rel_path="phase3/relabeling",
    filename_base="multi_class_labels",
    file_format=".pt",
    verbose=True
)
# Output: ✓ Saved: multi_class_labels_20260624_135247.pt

# Save numpy array
metrics_array = np.array([0.5, 0.6, 0.7])
path = manager.save_with_timestamp(
    metrics_array,
    rel_path="phase3/relabeling",
    filename_base="metrics",
    file_format=".npy",
    verbose=True
)

# Save JSON (dict)
config = {"pos_weight": 1.5, "epoch": 16}
path = manager.save_with_timestamp(
    config,
    rel_path="phase3/relabeling",
    filename_base="config",
    file_format=".json",
    verbose=True
)
```

### Retrieving Latest File

```python
# Get the most recent labels file
latest = manager.get_latest_file(
    rel_path="phase3/relabeling",
    filename_pattern="multi_class_labels"
)

if latest:
    print(f"Latest labels: {latest}")
    # Load it
    labels = torch.load(latest)
else:
    print("No saved labels found")
```

---

## Directory Structure

Files are saved to Google Drive following this structure:

```
MyDrive/
└── RETINNA_cache/
    ├── phase3/
    │   ├── relabeling/
    │   │   ├── multi_class_labels_20260624_135247.pt
    │   │   ├── multi_class_labels_20260624_140000.pt  ← timestamp
    │   │   ├── metrics_20260624_135247.json
    │   │   ├── config_20260624_135247.json
    │   │   └── qc_results_20260624_135247.json
    │   └── analysis/
    │       ├── tile_analysis_20260624_135247.json
    │       └── failure_cases_20260624_135247.pt
    └── phase4/  (future)
        └── rgb_ir_training/
```

---

## Handling Drive Issues

### Problem: Drive disconnects during long run

**Solution**: Test access periodically and reconnect if needed

```python
# In your training loop
if epoch % 5 == 0:  # Every 5 epochs
    if not manager.test_access(verbose=False):
        print(f"⚠️  Epoch {epoch}: Drive access lost, attempting reconnect...")
        if not manager.reconnect(max_attempts=2):
            print("✗ Could not reconnect. Saving to local and continuing...")
            # Fall back to local save
            torch.save(checkpoint, f"/content/temp_checkpoint_{epoch}.pt")
        else:
            print("✓ Reconnected, saving to Drive...")
            # Save to Drive
            manager.save_with_timestamp(checkpoint, "phase3/relabeling", f"checkpoint")
```

### Problem: Drive not accessible at start

**Solution**: Check and provide fallback

```python
drive_manager = setup_drive_for_colab(verbose=True)

if drive_manager is None:
    print("\n⚠️  WARNING: Using local-only mode (no Drive backup)")
    LOCAL_MODE = True
else:
    print("\n✓ Drive connected (timestamps will be saved)")
    LOCAL_MODE = False

# Use later:
if not LOCAL_MODE:
    manager.save_with_timestamp(data, "phase3/relabeling", "labels")
else:
    torch.save(data, "/content/local_labels.pt")
    print("⚠️  Saved locally only (no Drive backup)")
```

---

## Workflow for Phase 3 Notebook

### Setup Phase (Cell 1)

```python
import sys
sys.path.insert(0, '/content/RETINNA/src')
from drive_utils import setup_drive_for_colab

# Mount and test
drive_manager = setup_drive_for_colab(verbose=True)
assert drive_manager is not None, "Failed to connect to Google Drive"

print("\n" + "="*70)
print("PHASE 3: SPECTRAL RELABELING")
print("="*70)
print(f"Start time: {datetime.now().isoformat()}")
print(f"Drive status: {'✓ Connected' if drive_manager.test_access() else '✗ Disconnected'}")
```

### Processing Phase (Cells 2-N)

```python
# Before processing
print(f"\nStarting relabeling at {datetime.now().isoformat()}")

# ... your processing code ...

# After processing
print(f"Completed relabeling at {datetime.now().isoformat()}")

# Save with timestamp
output_path = drive_manager.save_with_timestamp(
    labels_array,
    rel_path="phase3/relabeling",
    filename_base="multi_class_labels",
    file_format=".pt"
)

print(f"✓ Saved to: {output_path}")
```

### Verification Phase (Cell N+1)

```python
# Verify Drive save
latest = manager.get_latest_file("phase3/relabeling", "multi_class_labels")
if latest:
    test_load = torch.load(latest)
    print(f"✓ Verified save: {latest}")
    print(f"  Shape: {test_load.shape}")
else:
    print("✗ Could not find saved labels")
```

---

## Environment-Specific Notes

### Local Machine (not Colab)

`DriveManager` will fail if not in Colab (no `google.colab` package).

Workaround:
```python
try:
    from drive_utils import setup_drive_for_colab
    drive_manager = setup_drive_for_colab()
except ImportError:
    print("Not in Colab, using local saves only")
    drive_manager = None
```

### Multiple Colab Notebooks Running

Each notebook gets its own mount point. Saves to Drive are atomic so concurrent writes are safe.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `"Permission denied" on Drive` | Re-authenticate: `python -m jupyter notebook --token=...` |
| `"Timeout" on Drive operations` | Try reconnect: `manager.reconnect()` |
| `"Drive already mounted"` | Use `force_remount=True` in `mount()` |
| `"Out of Drive space"` | Delete old timestamped files (older than 7 days) |
| Colab session keeps losing Drive | Restart runtime, remount at start of each cell block |

---

## Best Practices

1. **Test on startup**: Always run `setup_drive_for_colab()` at start
2. **Save periodically**: Don't wait until end of run to save (Colab sessions timeout)
3. **Keep timestamps**: They help track which run produced which file
4. **Monitor Drive space**: Old files accumulate; clean up monthly
5. **Verify saves**: Load saved files immediately to confirm success
6. **Log events**: Print when saving, loading, reconnecting (for debugging)

---

**Example Full Notebook Cell:**

```python
# ============================================================================
# SETUP: Google Drive and Phase 3 Configuration
# ============================================================================

import sys
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/content/RETINNA/src')
from drive_utils import setup_drive_for_colab
from dataset import get_dataloaders

# Mount Google Drive
print("📁 Initializing Google Drive...")
drive_manager = setup_drive_for_colab(verbose=True)

if drive_manager is None:
    raise RuntimeError("Failed to connect to Google Drive - aborting")

print("\n" + "="*70)
print("PHASE 3: SPECTRAL RELABELING WITH USGS MTBS STANDARDS")
print("="*70)
print(f"Session start: {datetime.now().isoformat()}")
print(f"Drive ready: {drive_manager.root}")

# Load dataset
print("\n📊 Loading CaBuAr dataset...")
dataloaders = get_dataloaders(batch_size=1, num_workers=0, normalize=True)
print(f"✓ Dataset loaded: {len(dataloaders['datasets']['train'])} train samples")

print("\n✓ Ready to begin Phase 3 relabeling")
```

---

**Status**: ✅ Ready for Phase 3 implementation  
**Author**: Stephen Cerruti with Claude Code  
**Date**: 2026-06-24
