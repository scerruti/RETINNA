# Paste as the LAST cell of notebooks/III_SWIR_NIR_CalFire_Segmentation.ipynb (in Colab),
# after the test-set evaluation cell. The notebook currently never saves its weights.
import torch, datetime

ckpt_path = f"/content/drive/MyDrive/RETINNA_DATA/phase3_nir_swir_{datetime.datetime.now():%Y%m%d_%H%M%S}.pt"
torch.save({
    "model_state_dict": model.state_dict(),          # best-by-val-F1 weights (restored in the test cell)
    "normalization": {"channel_means": channel_means.cpu(), "channel_stds": channel_stds.cpu()},
    "config": {
        "in_channels": 2, "bands": ["B08", "B12"], "band_indices_in_cabuar": [7, 11],
        "input_scale": "raw L2A DN, pre-04.00 baseline (no +1000 offset)",
        "post_fire_only": True, "tile": 512, "threshold_logit": 0.0,
    },
    "metrics": {"test_accuracy": test_accuracy, "test_precision": test_precision,
                "test_recall": test_recall, "test_f1": test_f1, "test_iou": test_iou},
    "split": {"train": list(train_dataset.indices), "val": list(val_dataset.indices),
              "test": list(test_dataset.indices), "fire_ids": fire_ids},
}, ckpt_path)
print("saved", ckpt_path)
