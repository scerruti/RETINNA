"""Endpoints and constants. Everything external the tool depends on lives here."""

import os
from pathlib import Path

# --- Imagery: Sentinel-2 L2A COGs on AWS (Element 84 Earth Search, us-west-2) ---
STAC_URL = os.getenv("BURNMAP_STAC_URL", "https://earth-search.aws.element84.com/v1")
S2_COLLECTION = "sentinel-2-l2a"
# Earth Search asset keys -> Sentinel-2 band names
ASSET_NIR = "nir"        # B08, 10 m
ASSET_SWIR = "swir22"    # B12, 20 m (resampled to 10 m, as in CaBuAr)
ASSET_RED = "red"        # B04, display only
ASSET_SCL = "scl"        # scene classification, cloud screening
# Processing baseline >= 04.00 (Jan 2022+) adds +1000 to L2A DN. CaBuAr training
# data predates that, so we subtract it to match the model's input distribution.
BOA_OFFSET = 1000
BOA_OFFSET_BASELINE = 4.0
# SCL classes treated as unusable: no data, saturated, cloud shadow, cloud med/high, cirrus
SCL_INVALID = (0, 1, 3, 8, 9, 10)

# --- Perimeters ---
# CAL FIRE FRAP historical perimeters (1878 through last calendar year)
FRAP_URL = (
    "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
    "California_Historic_Fire_Perimeters/FeatureServer/0/query"
)
# NIFC WFIGS interagency perimeters, current year to date (fills the gap until FRAP's annual release)
WFIGS_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Interagency_Perimeters_YearToDate/FeatureServer/0/query"
)

# --- Grid / inference ---
PIXEL_M = 10.0
SQM_PER_ACRE = 4046.8564224
TILE = 512
TILE_OVERLAP = 64
AOI_BUFFER_M = 2000.0          # pad around the perimeter so the model sees unburned context
MAX_AOI_PIXELS = 8000 * 8000   # ~6,400 km^2; refuse anything larger
DNBR_THRESHOLD = 0.10          # USGS "low severity" lower bound, used for the baseline map

# Model probability cutoff for "burned". Training used BCE pos_weight=2.0, which inflates
# probabilities, so 0.5 over-predicts; set this from scripts/threshold_sweep.py (multi-fire).
THRESHOLD = float(os.getenv("BURNMAP_THRESHOLD", "0.5"))

# --- Paths ---
MODEL_PATH = Path(os.getenv("BURNMAP_MODEL_PATH", "models/phase3_nir_swir.pt"))
RESULTS_DIR = Path(os.getenv("BURNMAP_RESULTS_DIR", "outputs"))
# Demo page. Defaults to the repo checkout; the container sets it (the package is installed
# into site-packages, so a path relative to this file would point at the wrong place).
WEB_DIR = Path(os.getenv("BURNMAP_WEB_DIR", Path(__file__).resolve().parents[2] / "web"))

# --- Deployment ---
# If set, finished results are published to s3://BUCKET/PREFIX<fire>/ and survive restarts;
# RESULTS_DIR becomes a local working copy. Unset = local disk only (dev).
RESULTS_BUCKET = os.getenv("BURNMAP_RESULTS_BUCKET", "")
RESULTS_PREFIX = os.getenv("BURNMAP_RESULTS_PREFIX", "results/")
# Public-demo guard rails: new (uncached) jobs are CPU-heavy and pull GBs of imagery.
MAX_ACTIVE_JOBS = int(os.getenv("BURNMAP_MAX_ACTIVE_JOBS", "3"))    # queued + running
NEW_JOBS_PER_HOUR = int(os.getenv("BURNMAP_NEW_JOBS_PER_HOUR", "10"))
# Lambda mode: both set by the CloudFormation stack. Unset = local dev (in-memory jobs, thread worker).
JOBS_TABLE = os.getenv("BURNMAP_JOBS_TABLE", "")
WORKER_FUNCTION = os.getenv("BURNMAP_WORKER_FUNCTION", "")
JOB_TIMEOUT_S = int(os.getenv("BURNMAP_JOB_TIMEOUT_S", "900"))  # Lambda maximum
# "cloudwatch" sends job metrics with PutMetricData; anything else only logs them as JSON lines.
METRICS_BACKEND = os.getenv("BURNMAP_METRICS", "log")
METRICS_NAMESPACE = os.getenv("BURNMAP_METRICS_NAMESPACE", "BurnScarMapper")
