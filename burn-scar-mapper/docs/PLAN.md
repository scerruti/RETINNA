# Burn Scar Mapper: plan and decisions

_Set up 2026-11-23._

## Status

| | Milestone | State |
|---|---|---|
| M0 | Scaffold: perimeter lookup, imagery, tiled inference, metrics, API, demo page, Dockerfile, tests | **Done.** Ran end to end on the Borel Fire (2024) with random weights in 4m44s from a home connection. |
| M1 | Real weights: re-run Phase III in Colab with `scripts/save_checkpoint_cell.py` | **Saved** 2026-11-24 to Drive as `RETINNA_DATA/phase3_nir_swir_20261124_032554.pt` (53.7 MB). **Inspected 2026-11-28:** passes the check. Test precision 0.68, recall 0.95, IoU 0.66, F1 0.79 (it over-predicts burned area). Test set shares no fire with training. |
| M2 | Validate on 4–6 fires; pick the threshold; choose featured fires | Borel 2024 done (2026-11-28): model IoU 0.65 (P 0.65, R 0.999) vs dNBR 0.86. Model over-predicts into dry grass and around clouds. Threshold sweep: IoU 0.92 at 0.95 (P 0.95, R 0.97), but that threshold was tuned on Borel itself; freeze it on other fires. Training used pos_weight=2.0, which pushes probabilities up. |
| M3 | Docker build and run locally | Needs Docker Desktop running |
| M4 | Deploy to AWS (us-west-2) | **Code done 2026-11-28 (serverless):** API + worker Lambdas, DynamoDB jobs, S3 results, alarms, budget alert, `deploy/`. Tests pass offline. Not deployed: needs Docker running, AWS CLI and credentials. |
| M5 | Polish: precomputed featured fires, README screenshot/GIF, resume link | |

## M1: Get a checkpoint you can stand behind

- Run the notebook and save with the cell in `scripts/`. Check that `channel_means` is on the raw-DN scale (NIR ≈ 2000–3500). If it's ≈ 0.2, the inputs were reflectance and `imagery._harmonize` must divide by 10,000.
- The notebook splits sequentially over sorted HDF5 keys (first 70% train, next 15% val, last 15% test), not at random. Leakage can only happen where a fire's patches straddle a split boundary. Check that when you write up results.
- The data cell now reads only B08/B12 from HDF5 (~1.1 GB). The old version held all 12 bands two or three times over and ran out of RAM on an L4 High-RAM runtime.
- The current notebook's markdown and `notebooks/phase3_inference.py` still say "NDVI + NIR". The trained model uses B08 + B12. The mapper follows the code, not the docs.

## M2: Validation protocol

CaBuAr covers fires from 2015–2022. The honest test is fires the model never saw:

1. Pick 2023–2025 fires of different sizes and vegetation (e.g. Park 2024, Borel 2024, Line 2024, plus a grass fire and a timber fire).
2. Pick at least one fire before 2022 as well. That exercises the path with no DN offset.
3. For each fire, record model vs perimeter and dNBR vs perimeter (both are in `result.json`).
4. Sweep `--threshold` (0.3–0.7) on 2–3 fires, then freeze it.

**Bar to beat:** on Borel 2024, plain dNBR already reaches IoU 0.86, precision 0.94 and recall 0.91 against the perimeter. Present the U-Net honestly against that bar. Ways it can win: no pre-fire scene needed (it uses post-fire only), and robustness when the pre-fire scene is poor.

## Decisions

| Decision | Why |
|---|---|
| **Earth Search on AWS** over Planetary Computer | Same cloud and region as the deployment (us-west-2), no request signing, COGs are directly range-readable. Planetary Computer would work as a fallback. |
| **Subtract 1000 DN when processing baseline ≥ 04.00** | Scenes from Jan 2022 onward carry a +1000 offset. CaBuAr's older scenes don't. Without this, every post-2022 input shifts by ~1 σ. Checked against Earth Search metadata (`raster:bands.offset = -0.1`, baseline 05.11). |
| **FRAP + WFIGS** | FRAP is CAL FIRE's official record but is released once a year (currently through 2025). WFIGS covers the current year. |
| **Lambda, not an always-on service** | A demo with occasional visitors should cost nothing when idle. App Runner is closed to new customers (2026-04-30), and ECS Express Mode keeps a task and a load balancer running (about $90/month). The worker gets 6 vCPU at 10 GB, more than a small Fargate task. Limit: 15 minutes per run, so precompute the largest fires. |
| **One image, two functions** | The API never imports torch, so its cold starts stay short. The worker reuses the model across warm invocations. arm64 is cheaper, and the image builds natively on Apple Silicon. |
| **DynamoDB for jobs** | Lambda invocations share nothing in memory. On-demand billing and a TTL keep it free and self-cleaning. |
| **Results in S3, layers via presigned redirects** | Fargate disk is ephemeral. The bucket stays private; the API checks file names, then redirects. |
| **Cache tied to checkpoint + threshold** | Changing the threshold after the multi-fire check must not keep serving maps made with the old one. |
| **Background jobs + polling** | A run takes minutes. That's longer than the request timeouts of App Runner, API Gateway and most proxies. |
| **dNBR shown next to the model** | It's the standard method, it's cheap, and it gives viewers (and interviewers) a reference point. |
| **Package lives in the RETINNA repo** | Keeps the model code and the notebook together. To publish it as a standalone portfolio repo later: `git subtree split -P burn-scar-mapper`. |

## M4: Deployment sketch (superseded: see README "Deploy to AWS" and `deploy/`)

- **ECR + App Runner** (us-west-2): simplest path for one container with HTTPS. Use 2 vCPU / 4 GB.
- App Runner disk is ephemeral. Before going public, move `RESULTS_DIR` to S3 (write `result.json` and layers on job completion; serve layers from S3/CloudFront) and keep the in-memory job table.
- Precompute 4–6 featured fires so most visitors get instant results. Rate-limit new jobs (e.g. 1 concurrent, N per hour) so a public page can't run up compute.
- Cost: App Runner bills provisioned memory while idle. Check current pricing. If idle cost matters, Lambda container images (15 min, 10 GB) with async invocation plus a static page on S3 is the near-zero-idle alternative.

## Known limitations / TODO

- **Speed:** `best_scene` loads each candidate scene at full resolution before checking clouds. Read SCL at 60 m first and only load the winner. Running in us-west-2 will also be much faster than from home.
- **Overlay projection:** UTM PNGs are draped on Web Mercator using lat/lon bounds, which skews them slightly on large fires. Fix by warping the PNGs to EPSG:3857, or serve tiles.
- **Acreage is clear-pixel acreage.** Clouds inside the perimeter are excluded; `perimeter_clear_fraction` reports coverage.
- **Perimeter ≠ burn map.** Unburned islands inside a perimeter count as misses. Say this whenever you quote recall.
- One worker, in-memory jobs, no auth or rate limit yet.
