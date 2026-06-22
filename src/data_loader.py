import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from torchgeo.datasets import CaBuAr


class CaBuArDataLoader:
    """Load and analyze the California Burned Areas (CaBuAr) dataset via TorchGeo."""

    def __init__(self, cache_dir: str = "./data", split: str = "test"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.split = split
        self.dataset = None
        self.stats = {}

    def load_dataset(self):
        """Load CaBuAr dataset using TorchGeo (handles HDF5 decompression internally)."""
        print(f"Loading CaBuAr dataset (split='{self.split}') via TorchGeo...")

        try:
            self.dataset = CaBuAr(root=str(self.cache_dir), download=True, split=self.split)
            print(f"✓ Dataset loaded: {len(self.dataset)} samples")
            return self.dataset
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            raise

    def compute_stats(self):
        """Compute basic statistics about the dataset."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        print("\nComputing dataset statistics...")

        total_samples = len(self.dataset)
        self.stats["total_samples"] = total_samples

        burned_pixels = 0
        unburned_pixels = 0

        for i in range(min(total_samples, 100)):  # Sample for efficiency
            sample = self.dataset[i]
            mask = sample['mask'].numpy().astype(int)
            burned_pixels += np.sum(mask > 0)
            unburned_pixels += np.sum(mask == 0)

        total_pixels = burned_pixels + unburned_pixels

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

    def visualize_samples(self, num_samples: int = 5):
        """Visualize sample tiles (pre-fire, post-fire, mask)."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        print(f"Visualizing {num_samples} sample tiles...\n")

        for idx in range(min(num_samples, len(self.dataset))):
            sample = self.dataset[idx]

            # Extract pre-fire and post-fire imagery (first 3 bands for RGB)
            image = sample['image'].numpy()  # shape: [2, 12, 512, 512]
            pre_fire = image[0, :3]  # shape: [3, 512, 512]
            post_fire = image[1, :3]
            mask = sample['mask'].numpy()[0]  # shape: [512, 512]

            # Normalize to 0-1 for visualization
            pre_fire_norm = (pre_fire - pre_fire.min()) / (pre_fire.max() - pre_fire.min() + 1e-8)
            post_fire_norm = (post_fire - post_fire.min()) / (post_fire.max() - post_fire.min() + 1e-8)

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Pre-fire image
            axes[0].imshow(np.transpose(pre_fire_norm, (1, 2, 0)))
            axes[0].set_title("Pre-Fire Image")
            axes[0].axis("off")

            # Post-fire image
            axes[1].imshow(np.transpose(post_fire_norm, (1, 2, 0)))
            axes[1].set_title("Post-Fire Image")
            axes[1].axis("off")

            # Ground truth mask
            axes[2].imshow(mask, cmap="RdYlGn_r")
            axes[2].set_title("Ground Truth Mask")
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()

            print(f"  Sample {idx}: pre-fire & post-fire imagery with burn mask")

        print()

    def save_stats(self, filepath: str = "./data_stats.json"):
        """Save statistics to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.stats, f, indent=2)
        print(f"Statistics saved to {filepath}")


if __name__ == "__main__":
    loader = CaBuArDataLoader()
    loader.load_dataset()
    loader.compute_stats()
    loader.visualize_samples(num_samples=3)
    loader.save_stats()
