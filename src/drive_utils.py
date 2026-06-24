"""
Google Drive utilities for Colab with error handling and reconnection logic.
Handles mounting, testing access, and timestamped saves.
"""

import os
import time
from datetime import datetime
from pathlib import Path
import subprocess
import json


class DriveManager:
    """Manage Google Drive mounting, access testing, and file operations."""

    def __init__(self, mount_point="/content/drive"):
        self.mount_point = mount_point
        self.root = Path(mount_point) / "MyDrive" / "RETINNA_cache"
        self.is_mounted = False
        self.last_access = None

    def mount(self, verbose=True):
        """Mount Google Drive to Colab."""
        if verbose:
            print("📁 Mounting Google Drive...")

        try:
            from google.colab import drive
            drive.mount(self.mount_point, force_remount=False)
            self.is_mounted = True
            if verbose:
                print(f"✓ Drive mounted at {self.mount_point}")
            return True
        except Exception as e:
            print(f"✗ Failed to mount Drive: {e}")
            self.is_mounted = False
            return False

    def unmount(self, verbose=True):
        """Unmount Google Drive."""
        if verbose:
            print("📁 Unmounting Google Drive...")

        try:
            subprocess.run(
                ["fusermount", "-u", self.mount_point],
                check=True,
                capture_output=True
            )
            self.is_mounted = False
            if verbose:
                print(f"✓ Drive unmounted")
            return True
        except Exception as e:
            print(f"⚠ Unmount may have failed: {e}")
            self.is_mounted = False
            return False

    def test_access(self, verbose=True):
        """Test if Google Drive is accessible."""
        if not self.is_mounted:
            if verbose:
                print("✗ Drive not mounted")
            return False

        try:
            # Try to list directory
            test_path = Path(self.mount_point) / "MyDrive"
            contents = list(test_path.iterdir())
            self.last_access = datetime.now()

            if verbose:
                print(f"✓ Drive access OK ({len(contents)} items in MyDrive)")
            return True
        except Exception as e:
            if verbose:
                print(f"✗ Drive access failed: {e}")
            return False

    def reconnect(self, max_attempts=3, verbose=True):
        """Attempt to reconnect to Google Drive."""
        if verbose:
            print(f"🔄 Attempting to reconnect to Google Drive...")

        for attempt in range(max_attempts):
            if verbose:
                print(f"  Attempt {attempt + 1}/{max_attempts}")

            # Try unmounting
            self.unmount(verbose=False)
            time.sleep(2)

            # Try mounting
            if self.mount(verbose=False):
                time.sleep(2)

                # Test access
                if self.test_access(verbose=False):
                    if verbose:
                        print(f"✓ Reconnected successfully")
                    return True

            if attempt < max_attempts - 1:
                time.sleep(3)

        if verbose:
            print(f"✗ Failed to reconnect after {max_attempts} attempts")
        return False

    def ensure_directory(self, rel_path, verbose=True):
        """Ensure directory exists in Drive."""
        dir_path = self.root / rel_path

        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"✓ Directory ready: {rel_path}")
            return dir_path
        except Exception as e:
            print(f"✗ Failed to create directory {rel_path}: {e}")
            return None

    def save_with_timestamp(self, data, rel_path, filename_base,
                           file_format=".pt", verbose=True):
        """
        Save data with timestamp.

        Args:
            data: Data to save (torch tensor, numpy array, dict, etc)
            rel_path: Relative path in Drive (e.g., "phase3/relabeling")
            filename_base: Base filename (e.g., "labels")
            file_format: File extension (e.g., ".pt", ".json", ".png")
            verbose: Print status

        Returns:
            Path to saved file if successful, None otherwise
        """
        import torch
        import numpy as np

        # Ensure directory exists
        dir_path = self.ensure_directory(rel_path, verbose=False)
        if dir_path is None:
            print(f"✗ Failed to ensure directory {rel_path}")
            return None

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_base}_{timestamp}{file_format}"
        full_path = dir_path / filename

        try:
            if file_format == ".pt":
                torch.save(data, full_path)
            elif file_format == ".json":
                with open(full_path, 'w') as f:
                    json.dump(data, f, indent=2)
            elif file_format == ".npy":
                np.save(full_path, data)
            elif file_format == ".png":
                # Assume data is PIL Image or similar
                data.save(full_path)
            else:
                # Try generic save
                with open(full_path, 'wb') as f:
                    if hasattr(data, 'save'):
                        data.save(f)
                    else:
                        raise ValueError(f"Don't know how to save {file_format}")

            if verbose:
                print(f"✓ Saved: {filename}")

            return full_path

        except Exception as e:
            print(f"✗ Failed to save {filename}: {e}")
            return None

    def get_latest_file(self, rel_path, filename_pattern):
        """Get the most recently saved file matching pattern."""
        dir_path = self.root / rel_path

        if not dir_path.exists():
            return None

        matching_files = list(dir_path.glob(f"{filename_pattern}_*"))

        if not matching_files:
            return None

        # Sort by modification time, return newest
        return sorted(matching_files, key=lambda p: p.stat().st_mtime)[-1]


def setup_drive_for_colab(verbose=True):
    """
    Quick setup: mount drive, test access, ensure RETINNA cache exists.
    Returns DriveManager instance.
    """
    manager = DriveManager()

    if not manager.mount(verbose=verbose):
        return None

    time.sleep(2)

    if not manager.test_access(verbose=verbose):
        print("⚠ Initial access test failed, attempting reconnect...")
        if not manager.reconnect(verbose=verbose):
            return None

    # Ensure cache directory exists
    manager.ensure_directory("", verbose=verbose)

    return manager
