"""fire id -> imagery -> model -> burn map, acreage, and perimeter comparison."""

import json
import logging
import time
from pathlib import Path

import numpy as np
from pyproj import CRS
from shapely.geometry import box, mapping

from . import analysis, config, imagery, inference, perimeters, render
from .model import LoadedModel

log = logging.getLogger(__name__)


def _geojson(geoms, crs) -> dict:
    to_ll = lambda g: imagery.to_crs(g, CRS.from_epsg(4326), crs)  # noqa: E731
    return {"type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}, "geometry": mapping(to_ll(g))} for g in geoms]}


def run(fire_id: str, model: LoadedModel, out_root: Path = config.RESULTS_DIR, threshold: float = 0.5) -> dict:
    t0 = time.time()
    out = Path(out_root) / fire_id.replace(":", "_")
    out.mkdir(parents=True, exist_ok=True)

    fire = perimeters.get(fire_id)
    if fire.geometry is None or fire.start is None:
        raise ValueError(f"{fire_id} has no perimeter geometry or start date")
    log.info("fire %s %s (%s ac)", fire.name, fire.year, fire.acres)

    pre, post = imagery.fire_scenes(fire.geometry, fire.start, fire.end)
    assert pre.nir.shape == post.nir.shape, "pre/post grids differ"
    log.info("scenes: pre %s (%.0f%% clear), post %s (%.0f%% clear)",
             pre.date, 100 * pre.valid_fraction, post.date, 100 * post.valid_fraction)

    # --- model ---
    x = inference.normalize(post.nir, post.swir, model.channel_means, model.channel_stds)
    prob = inference.predict_probabilities(model.net, x, device=model.device)
    valid = post.valid
    pred = (prob > threshold) & valid

    # --- references ---
    perim_utm = imagery.to_crs(fire.geometry, post.crs)
    ref = analysis.rasterize(perim_utm, pred.shape, post.transform)
    dnbr_pred, _ = analysis.dnbr_mask(pre.nir, pre.swir, post.nir, post.swir)
    both_valid = valid & pre.valid
    dnbr_pred &= both_valid

    model_vs_perim = analysis.compare(pred, ref, valid)
    dnbr_vs_perim = analysis.compare(dnbr_pred, ref, both_valid)

    # --- artifacts ---
    render.false_color(post.swir, post.nir, post.red, valid).save(out / "post_false_color.jpg", quality=85)
    render.false_color(pre.swir, pre.nir, pre.red, pre.valid).save(out / "pre_false_color.jpg", quality=85)
    render.mask_overlay(pred).save(out / "model_burn.png")
    render.probability_overlay(np.where(valid, prob, 0)).save(out / "model_probability.png")
    render.mask_overlay(dnbr_pred, color=(255, 220, 0)).save(out / "dnbr_burn.png")
    polys = analysis.vectorize(pred, post.transform)
    (out / "model_burn.geojson").write_text(json.dumps(_geojson(polys, post.crs)))
    (out / "perimeter.geojson").write_text(json.dumps(_geojson([perim_utm], post.crs)))

    h, w = pred.shape
    (x0, y0), (x1, y1) = post.transform * (0, 0), post.transform * (w, h)
    grid_box = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    bounds_ll = imagery.to_crs(grid_box, CRS.from_epsg(4326), post.crs).bounds

    result = {
        "fire": fire.summary(),
        "scenes": {
            "pre": {"date": pre.date.isoformat(), "items": pre.item_ids, "clear_fraction": round(pre.valid_fraction, 3)},
            "post": {"date": post.date.isoformat(), "items": post.item_ids, "clear_fraction": round(post.valid_fraction, 3)},
        },
        "model": {"threshold": threshold, **model_vs_perim.to_dict()},
        "dnbr_baseline": {"threshold": config.DNBR_THRESHOLD, **dnbr_vs_perim.to_dict()},
        "official_acres": round(fire.acres, 1),
        # Share of the perimeter with usable (cloud-free) post-fire pixels; acreage and
        # agreement are computed over clear pixels only.
        "perimeter_clear_fraction": round(float(valid[ref].mean()) if ref.any() else 0.0, 3),
        "grid": {"crs": post.crs.to_string(), "shape": [h, w], "pixel_m": config.PIXEL_M,
                 "bounds_4326": [round(v, 6) for v in bounds_ll]},
        "layers": {"post_false_color": "post_false_color.jpg", "pre_false_color": "pre_false_color.jpg",
                   **{k: f"{k}.png" for k in ("model_burn", "model_probability", "dnbr_burn")}},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return result
