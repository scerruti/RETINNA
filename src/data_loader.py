import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from pathlib import Path
import json


class CaBuArDataLoader:
    """Load and analyze the California Burned Areas (CaBuAr) dataset."""

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.dataset = None
        self.stats = {}

    def load_dataset(self):
        """Download and load the CaBuAr dataset from Hugging Face."""
        print("Loading CaBuAr dataset from Hugging Face...")
        self.dataset = load_dataset(
            "DarthReca/california_burned_areas",
            cache_dir=str(self.cache_dir),
            trust_remote_code=True
        )
        print(f"Dataset loaded: {self.dataset}")
        return self.dataset

    def compute_stats(self):
        """Compute basic statistics about the dataset."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        print("\nComputing dataset statistics...")

        total_samples = len(self.dataset["train"])
        self.stats["total_samples"] = total_samples

        # Analyze class balance
        burned_pixels = 0
        unburned_pixels = 0

        for sample in self.dataset["train"]:
            mask = np.array(sample["mask"])
            burned_pixels += np.sum(mask > 0)
            unburned_pixels += np.sum(mask == 0)

        total_pixels = burned_pixels + unburned_pixels
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

        for idx in range(min(num_samples, len(self.dataset["train"]))):
            sample = self.dataset["train"][idx]

            # Extract images and mask
            pre_fire = np.array(sample["pre_fire"])
            post_fire = np.array(sample["post_fire"])
            mask = np.array(sample["mask"])

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Pre-fire image (normalize to 0-1 for visualization)
            pre_fire_normalized = (pre_fire - pre_fire.min()) / (pre_fire.max() - pre_fire.min())
            axes[0].imshow(pre_fire_normalized)
            axes[0].set_title("Pre-Fire Image")
            axes[0].axis("off")

            # Post-fire image
            post_fire_normalized = (post_fire - post_fire.min()) / (post_fire.max() - post_fire.min())
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
