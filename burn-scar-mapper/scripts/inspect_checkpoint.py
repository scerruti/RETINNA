"""python scripts/inspect_checkpoint.py path/to/model.pt

Tells you whether a checkpoint is the 2-channel NIR+SWIR model the mapper needs."""

import sys

import torch

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
w = sd.get("inc.double_conv.0.weight")
print("keys:", list(ckpt)[:10] if isinstance(ckpt, dict) else type(ckpt))
print("first conv:", tuple(w.shape) if w is not None else "not found (different architecture)")
print("output conv:", tuple(sd["outc.weight"].shape) if "outc.weight" in sd else "n/a")
print("normalization:", ckpt.get("normalization"))
print("metrics:", ckpt.get("metrics"))
if w is not None:
    ok = w.shape[1] == 2 and sd["outc.weight"].shape[0] == 1 and "normalization" in ckpt
    print("\nUSABLE BY BURN SCAR MAPPER" if ok else "\nNOT the Phase III NIR+SWIR model")
