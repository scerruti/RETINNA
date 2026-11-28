"""burnmap search PARK --year 2024
burnmap run FRAP:12345 --checkpoint models/phase3_nir_swir.pt"""

import argparse
import json
import logging
from pathlib import Path

from . import config, perimeters, pipeline
from .model import load_checkpoint


def main(argv=None):
    ap = argparse.ArgumentParser(prog="burnmap")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search", help="find a fire by name")
    s.add_argument("name")
    s.add_argument("--year", type=int)
    s.add_argument("--min-acres", type=float, default=1000)
    r = sub.add_parser("run", help="map one fire")
    r.add_argument("fire_id", help="e.g. FRAP:12345 (from `burnmap search`)")
    r.add_argument("--checkpoint", type=Path, default=config.MODEL_PATH)
    r.add_argument("--out", type=Path, default=config.RESULTS_DIR)
    r.add_argument("--threshold", type=float, default=config.THRESHOLD)
    r.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("rasterio").setLevel(logging.WARNING)

    if a.cmd == "search":
        for f in perimeters.search(a.name, a.year, a.min_acres):
            s = f.summary()
            print(f"{s['id']:<16} {s['name']:<24} {s['year']}  {s['acres']:>11,.0f} ac  {s['start']} -> {s['end']}")
    else:
        model = load_checkpoint(a.checkpoint, a.device)
        result = pipeline.run(a.fire_id, model, a.out, a.threshold)
        print(json.dumps({k: result[k] for k in ("fire", "scenes", "model", "dnbr_baseline", "elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()
