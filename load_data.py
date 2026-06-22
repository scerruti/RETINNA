#!/usr/bin/env python3
"""
Data Loading Script for CaBuAr Dataset
Run this script to download, analyze, and visualize the dataset.
"""

from src.data_loader import CaBuArDataLoader
import argparse


def main():
    parser = argparse.ArgumentParser(description="Load and analyze CaBuAr dataset")
    parser.add_argument("--cache-dir", default="./data", help="Directory to cache dataset")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--viz-dir", default="./visualizations", help="Directory to save visualizations")
    parser.add_argument("--stats-file", default="./data_stats.json", help="JSON file to save stats")

    args = parser.parse_args()

    loader = CaBuArDataLoader(cache_dir=args.cache_dir)
    loader.load_dataset()
    loader.compute_stats()
    loader.visualize_samples(num_samples=args.num_samples, save_dir=args.viz_dir)
    loader.save_stats(filepath=args.stats_file)

    print("✓ Data loading complete!")


if __name__ == "__main__":
    main()
