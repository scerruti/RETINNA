import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import h5py
import urllib.request
import os


class CaBuArDataLoader:
    """Load and analyze the California Burned Areas (CaBuAr) dataset."""

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.dataset = None
        self.stats = {}

    def load_dataset(self, num_files: int = 5):
        """Download and load CaBuAr dataset HDF5 files from Hugging Face (raw URLs)."""
        print(f"Loading CaBuAr dataset ({num_files} HDF5 files) from Hugging Face...")

        self.dataset = []
        base_url = "https://huggingface.co/datasets/DarthReca/california_burned_areas/resolve/main/normalized/complete/"

        # Download and load individual HDF5 files
        for i in range(1, num_files + 1):
            filename = f"california_{i}.hdf5"
            try:
                print(f"  Downloading {filename}...")
                url = base_url + filename
                cache_path = self.cache_dir / filename

                # Download file if not already cached
                if not cache_path.exists():
                    urllib.request.urlretrieve(url, cache_path)
                    print(f"    Downloaded to {cache_path}")
                else:
                    print(f"    Using cached {cache_path}")

                # Load and parse HDF5 file
                with h5py.File(cache_path, 'r') as h5_file:
                    samples = self._parse_hdf5_file(h5_file)
                    self.dataset.extend(samples)
                    print(f"    Loaded {len(samples)} samples")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
                break

        print(f"Total dataset size: {len(self.dataset)} samples")
        return self.dataset

    def _parse_hdf5_file(self, h5_file):
        """Parse single HDF5 file into list of samples."""
        samples = []

        for wildfire_id in h5_file.keys():
            group = h5_file[wildfire_id]

            # Extract pre-fire, post-fire, and mask
            pre_fire = np.array(group['pre_fire'])
            post_fire = np.array(group['post_fire'])
            mask = np.array(group['mask'])

            samples.append({
                'wildfire_id': wildfire_id,
                'pre_fire': pre_fire,
                'post_fire': post_fire,
                'mask': mask
            })

        return samples

    def compute_stats(self):
        """Compute basic statistics about the dataset."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        print("\nComputing dataset statistics...")

        total_samples = len(self.dataset)
        self.stats["total_samples"] = total_samples

        # Analyze class balance
        burned_pixels = 0
        unburned_pixels = 0

        for sample in self.dataset:
            mask = np.array(sample["mask"])
            burned_pixels += np.sum(mask > 0)
            unburned_pixels += np.sum(mask == 0)

        total_pixels = burned_pixels + unburned_pixels

        # Guard against divide by zero
        if total_pixels == 0:
            burned_percent = 0.0
            unburned_percent = 0.0
            print("Warning: No pixels found in masks. Check data integrity.")
        else:
            burned_percent = (burned_pixels / total_pixels) * 100
            unburned_percent = (unburned_pixels / total_pixels) * 100

        self.stats["total_pixels"] = int(total_pixels)
        self.stats["burned_pixels"] = int(burned_pixels)
        self.stats["unburned_pixels"] = int(unburned_pixels)
        self.stats["burned_percent"] = float(burned_percent)
        self.stats["unburned_percent"] = float(unburned_percent)

        self._print_stats()
        return self.stats

    def _print_stats(self):
        """Print statistics to console."""
        print(f"\n{'='*50}")
        print("CaBuAr Dataset Statistics")
        print(f"{'='*50}")
        print(f"Total samples: {self.stats['total_samples']:,}")
        print(f"Total pixels: {self.stats['total_pixels']:,}")
        print(f"Burned pixels: {self.stats['burned_pixels']:,} ({self.stats['burned_percent']:.2f}%)")
        print(f"Unburned pixels: {self.stats['unburned_pixels']:,} ({self.stats['unburned_percent']:.2f}%)")
        print(f"{'='*50}\n")

    def visualize_samples(self, num_samples: int = 5, save_dir: str = "./visualizations"):
        """Visualize and save sample tiles."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        print(f"Visualizing {num_samples} sample tiles...")

        for idx in range(min(num_samples, len(self.dataset))):
            sample = self.dataset[idx]

            # Extract images and mask (use first 3 channels for RGB visualization)
            pre_fire = np.array(sample["pre_fire"])
            post_fire = np.array(sample["post_fire"])
            mask = np.array(sample["mask"])

            # Use first 3 channels if multi-channel
            if pre_fire.ndim == 3 and pre_fire.shape[0] > 3:
                pre_fire_viz = pre_fire[:3].transpose(1, 2, 0)
                post_fire_viz = post_fire[:3].transpose(1, 2, 0)
            else:
                pre_fire_viz = pre_fire if pre_fire.ndim == 2 else pre_fire[0]
                post_fire_viz = post_fire if post_fire.ndim == 2 else post_fire[0]

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Pre-fire image (normalize to 0-1 for visualization)
            pre_fire_normalized = (pre_fire_viz - np.min(pre_fire_viz)) / (np.max(pre_fire_viz) - np.min(pre_fire_viz) + 1e-8)
            axes[0].imshow(pre_fire_normalized)
            axes[0].set_title("Pre-Fire Image")
            axes[0].axis("off")

            # Post-fire image
            post_fire_normalized = (post_fire_viz - np.min(post_fire_viz)) / (np.max(post_fire_viz) - np.min(post_fire_viz) + 1e-8)
            axes[1].imshow(post_fire_normalized)
            axes[1].set_title("Post-Fire Image")
            axes[1].axis("off")

            # Ground truth mask
            axes[2].imshow(mask, cmap="RdYlGn_r")
            axes[2].set_title("Ground Truth Mask")
            axes[2].axis("off")

            plt.tight_layout()
            filepath = save_path / f"sample_{idx:03d}.png"
            plt.savefig(filepath, dpi=100, bbox_inches="tight")
            plt.close()

            print(f"  Saved: {filepath}")

        print(f"Visualizations saved to {save_path}\n")

    def save_stats(self, filepath: str = "./data_stats.json"):
        """Save statistics to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.stats, f, indent=2)
        print(f"Statistics saved to {filepath}")


if __name__ == "__main__":
    loader = CaBuArDataLoader()
    loader.load_dataset()
    loader.compute_stats()
    loader.visualize_samples(num_samples=5)
    loader.save_stats()
