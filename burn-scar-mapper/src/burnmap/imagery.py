"""Find and load cloud-free Sentinel-2 L2A scenes for a fire from AWS Earth Search."""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pystac
import pystac_client
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.stac import load as stac_load
from pyproj import CRS, Transformer
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform

from . import config


@dataclass
class Scene:
    date: date
    item_ids: list[str]
    nir: np.ndarray      # B08, harmonized DN (pre-04.00 scale), 0 = nodata
    swir: np.ndarray     # B12 resampled to 10 m
    red: np.ndarray      # B04, display only
    valid: np.ndarray    # bool: has data and not cloud/shadow per SCL
    transform: Affine
    crs: CRS

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean())


def utm_crs_for(geom: BaseGeometry) -> CRS:
    lon, lat = geom.centroid.x, geom.centroid.y
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def to_crs(geom: BaseGeometry, dst: CRS, src: CRS = CRS.from_epsg(4326)) -> BaseGeometry:
    t = Transformer.from_crs(src, dst, always_xy=True)
    return shp_transform(t.transform, geom)


def search(bbox_4326, start: date, end: date, max_cloud: float = 40) -> list[pystac.Item]:
    client = pystac_client.Client.open(config.STAC_URL)
    items = client.search(
        collections=[config.S2_COLLECTION], bbox=bbox_4326,
        datetime=f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
        query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=200,
    ).item_collection()
    return list(items)


def _group_by_day(items: list[pystac.Item]) -> dict[date, list[pystac.Item]]:
    groups: dict[date, list[pystac.Item]] = {}
    for it in items:
        groups.setdefault(it.datetime.date(), []).append(it)
    return groups


def _harmonize(dn: np.ndarray, baseline: float) -> np.ndarray:
    """Remove the +1000 BOA offset added from processing baseline 04.00 onward."""
    dn = dn.astype(np.float32)
    if baseline >= config.BOA_OFFSET_BASELINE:
        nodata = dn == 0
        dn = np.clip(dn - config.BOA_OFFSET, 0, None)
        dn[nodata] = 0
    return dn


def _grid(aoi_utm: BaseGeometry, crs: CRS) -> GeoBox:
    """Same AOI -> same pixel grid, so pre and post scenes align exactly."""
    return GeoBox.from_bbox(aoi_utm.bounds, crs=crs.to_string(), resolution=config.PIXEL_M)


def load_scene(items: list[pystac.Item], aoi_utm: BaseGeometry, crs: CRS) -> Scene:
    minx, miny, maxx, maxy = aoi_utm.bounds
    n_px = ((maxx - minx) / config.PIXEL_M) * ((maxy - miny) / config.PIXEL_M)
    if n_px > config.MAX_AOI_PIXELS:
        raise ValueError(f"AOI is {n_px / 1e6:.0f} Mpx; limit is {config.MAX_AOI_PIXELS / 1e6:.0f} Mpx")
    ds = stac_load(
        items,
        bands=[config.ASSET_NIR, config.ASSET_SWIR, config.ASSET_RED, config.ASSET_SCL],
        geobox=_grid(aoi_utm, crs),
        groupby="solar_day", chunks=None,
        resampling={config.ASSET_SWIR: "bilinear", config.ASSET_SCL: "nearest", "*": "nearest"},
    ).isel(time=0)
    baseline = float(items[0].properties.get("s2:processing_baseline", "0") or 0)
    nir = _harmonize(ds[config.ASSET_NIR].values, baseline)
    swir = _harmonize(ds[config.ASSET_SWIR].values, baseline)
    red = _harmonize(ds[config.ASSET_RED].values, baseline)
    scl = ds[config.ASSET_SCL].values
    valid = (nir > 0) & (swir > 0) & ~np.isin(scl, config.SCL_INVALID)
    return Scene(
        date=items[0].datetime.date(), item_ids=[i.id for i in items], nir=nir, swir=swir, red=red,
        valid=valid, transform=ds.odc.geobox.affine, crs=crs,
    )


def best_scene(
    aoi_4326: BaseGeometry, start: date, end: date, prefer: str, min_valid: float = 0.85,
    max_tries: int = 6,
) -> Scene:
    """Pick the scene closest to the fire ('latest' before it, 'earliest' after it)
    whose AOI is at least `min_valid` cloud-free; fall back to the clearest one tried."""
    crs = utm_crs_for(aoi_4326)
    aoi_utm = to_crs(aoi_4326, crs).buffer(config.AOI_BUFFER_M).envelope
    bbox = to_crs(aoi_utm, CRS.from_epsg(4326), crs).bounds
    groups = _group_by_day(search(bbox, start, end))
    if not groups:
        raise LookupError(f"No Sentinel-2 scenes between {start} and {end}")
    days = sorted(groups, reverse=(prefer == "latest"))
    best = None
    for d in days[:max_tries]:
        scene = load_scene(groups[d], aoi_utm, crs)
        if scene.valid_fraction >= min_valid:
            return scene
        if best is None or scene.valid_fraction > best.valid_fraction:
            best = scene
    return best


def fire_scenes(perimeter_4326: BaseGeometry, fire_start: date, fire_end: date | None):
    """Pre-fire: up to 60 days before ignition. Post-fire: from containment (or +30 days
    after ignition for active fires) up to 90 days later."""
    pre = best_scene(perimeter_4326, fire_start - timedelta(days=60), fire_start - timedelta(days=1), "latest")
    post_start = (fire_end or fire_start + timedelta(days=30)) + timedelta(days=1)
    post_end = min(post_start + timedelta(days=90), date.today())
    post = best_scene(perimeter_4326, post_start, post_end, "earliest")
    return pre, post
