# Burn Scar Mapper: plan and decisions

_Set up 2026-11-23._

## Status

| | Milestone | State |
|---|---|---|
| M0 | Scaffold: perimeter lookup, imagery, tiled inference, metrics, API, demo page, Dockerfile, tests | **Done.** Ran end to end on the Borel Fire (2024) with random weights in 4m44s from a home connection. |
| M1 | Real weights: re-run Phase III in Colab with `scripts/save_checkpoint_cell.py` | **Blocking.** No saved NIR+SWIR checkpoint exists. |
| M2 | Validate on 4–6 fires; pick the threshold; choose featured fires | |
| M3 | Docker build and run locally | Needs Docker Desktop running |
| M4 | Deploy to AWS (us-west-2) with an S3 results cache | |
| M5 | Polish: precomputed featured fires, README screenshot/GIF, resume link | |

## M1: Get a checkpoint you can stand behind

- Run the notebook and save with the cell in `scripts/`. Check that `channel_means` is on the raw-DN scale (NIR ≈ 2000–3500). If it's ≈ 0.2, the inputs were reflectance and `imagery._harmonize` must divide by 10,000.
- **Consider splitting by fire, not by patch, when you retrain.** The notebook splits the 534 patches at random. Neighboring patches from the same fire can land in both train and test, which makes the 0.79 F1 optimistic. A fire-level split gives a number you can defend in an interview.
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
| **Background jobs + polling** | A run takes minutes. That's longer than the request timeouts of App Runner, API Gateway and most proxies. |
| **dNBR shown next to the model** | It's the standard method, it's cheap, and it gives viewers (and interviewers) a reference point. |
| **Package lives in the RETINNA repo** | Keeps the model code and the notebook together. To publish it as a standalone portfolio repo later: `git subtree split -P burn-scar-mapper`. |

## M4: Deployment sketch

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
