# Burn Scar Mapper

Pick a California wildfire. The tool pulls free Sentinel-2 imagery from before and after the fire, runs a U-Net trained on the CaBuAr dataset, and returns:

- a burn map (PNG overlay and GeoJSON polygons)
- burned acreage
- agreement with the official perimeter (IoU, precision, recall), side by side with a classic dNBR baseline

It's a FastAPI service in a Docker container with a one-page Leaflet demo. The model comes from Phase III of [RETINNA](../README.md) (`notebooks/III_SWIR_NIR_CalFire_Segmentation.ipynb`).

## How it works

```
fire name ──► CAL FIRE FRAP (≤ last year) / NIFC WFIGS (this year) ──► perimeter + dates
                                │
                                ▼
         Earth Search STAC (AWS, us-west-2): Sentinel-2 L2A COGs
         pre-fire: latest clear scene ≤ 60 days before ignition
         post-fire: earliest clear scene ≤ 90 days after containment
                                │   B08 (NIR) + B12 (SWIR→10 m) + B04 + SCL cloud mask
                                │   harmonize: subtract +1000 DN offset (baseline ≥ 04.00)
                                ▼
         U-Net (2-ch post-fire NIR+SWIR, z-scored) on 512×512 tiles, 64 px overlap, blended
                                │
                                ▼
         burn mask ─► acres, IoU/precision/recall vs perimeter ─► PNG/GeoJSON layers
         dNBR (pre−post NBR > 0.10) computed on the same grid as a baseline
```

## Quick start

```bash
cd burn-scar-mapper
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -m "not network"            # offline tests
.venv/bin/pytest -m network                  # live FRAP lookup

.venv/bin/burnmap search Park --year 2024
.venv/bin/burnmap run FRAP:<id> --checkpoint models/phase3_nir_swir.pt

.venv/bin/uvicorn burnmap.api:app --reload   # http://localhost:8000
```

Docker (the image targets AWS Lambda; for a local run, override the entrypoint to use uvicorn):

```bash
docker build --build-arg MODEL_FILE=phase3_nir_swir_20261124_032554.pt -t burn-scar-mapper .
docker run -p 8000:8000 --entrypoint uvicorn burn-scar-mapper burnmap.api:app --host 0.0.0.0
```

## Deploy to AWS (serverless)

`deploy/deploy.sh models/<checkpoint>.pt` builds the arm64 image, pushes it to ECR, and creates or updates the CloudFormation stack in [`deploy/cloudformation.yaml`](deploy/cloudformation.yaml) (us-west-2). Nothing runs while nobody is using it.

```
browser ──► API Lambda (function URL) ──► DynamoDB jobs table
               │  page, search, job status,          ▲
               │  presigned links to S3 layers        │ status / result
               └─ async invoke ──► worker Lambda ─────┘
                                   (10 GB / 6 vCPU, 15 min max)
                                   Sentinel-2 COGs ─► U-Net + dNBR ─► S3 results
```

- **One image, two functions.** The API function never loads the model or imports torch, so its cold starts stay short. The worker loads the model once per cold start and maps one fire per invocation. Failed runs are reported, not retried.
- **Results in a private S3 bucket**, served through short-lived presigned redirects. A stored result is reused only if it came from the same checkpoint and threshold.
- **Jobs in DynamoDB** (on-demand billing, 7-day TTL). A job killed at the 15-minute limit is reported as timed out.
- **Monitoring:** custom metrics `JobSucceeded`, `JobFailed`, `JobDuration`, `CacheHit`, `RateLimited`; alarms on failed jobs, worker crashes (timeout or out-of-memory) and API errors; a dashboard; optional email alerts and a monthly budget alert.
- **Guard rails:** at most 3 active jobs and 10 new fires per hour (application limits), plus optional reserved concurrency on the worker as a hard cap.

Settings are environment variables: `THRESHOLD`, `NEW_JOBS_PER_HOUR`, `WORKER_MEMORY_MB`, `WORKER_CONCURRENCY`, `ALARM_EMAIL`, `MONTHLY_BUDGET_USD`. Cost and teardown are noted at the top of `deploy.sh`. Very large fires (Park 2024, about 430k acres) may not finish within Lambda's 15 minutes; precompute those.

## Get the weights

The weights aren't in git, and **the Phase III notebook doesn't save them yet**. To produce a checkpoint:

1. Open `notebooks/III_SWIR_NIR_CalFire_Segmentation.ipynb` in Colab and run it.
2. Paste [`scripts/save_checkpoint_cell.py`](scripts/save_checkpoint_cell.py) as the last cell and run it. It saves the weights, normalization stats, config, metrics and split to Drive.
3. Download the file to `models/phase3_nir_swir.pt` and check it:
   `.venv/bin/python scripts/inspect_checkpoint.py models/phase3_nir_swir.pt`

The `unet_model_*.pt` files on Drive come from Phase II (4/8-channel inputs, 7-class output). The loader rejects them.

## Layout

| Path | What |
|---|---|
| `src/burnmap/perimeters.py` | Fire search + perimeter geometry (FRAP, WFIGS) |
| `src/burnmap/imagery.py` | STAC search, cloud screening, scene loading, DN harmonization |
| `src/burnmap/model.py` | U-Net (key-compatible with the notebook), checkpoint loader |
| `src/burnmap/inference.py` | Tiled, blended sliding-window inference |
| `src/burnmap/analysis.py` | Acreage, perimeter agreement, dNBR, vectorization |
| `src/burnmap/pipeline.py` | End-to-end run → `outputs/<fire>/result.json` + layers |
| `src/burnmap/api.py` | FastAPI: search, jobs, stored results, demo page |
| `src/burnmap/lambda_api.py`, `worker.py` | Lambda handlers: API (via Mangum) and the mapping worker |
| `src/burnmap/jobs.py`, `storage.py`, `metrics.py` | Job records (memory / DynamoDB) and dispatch, results (disk / S3), metrics |
| `deploy/` | CloudFormation stack and deploy script |
| `web/index.html` | Demo page |
| `docs/PLAN.md` | Build plan, decisions, known risks |

## Data sources

- Sentinel-2 L2A: [Element 84 Earth Search](https://earth-search.aws.element84.com/v1) (AWS Open Data, no auth)
- Perimeters: [CAL FIRE FRAP California Fire Perimeters](https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/California_Historic_Fire_Perimeters/FeatureServer/0) and [NIFC WFIGS Interagency Perimeters (year to date)](https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters_YearToDate/FeatureServer/0)
- Training data: [CaBuAr](https://huggingface.co/datasets/DarthReca/california_burned_areas)

MIT License.
