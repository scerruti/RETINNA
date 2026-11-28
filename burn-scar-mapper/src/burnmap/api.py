"""FastAPI service.

GET  /api/health                 mode, checkpoint, threshold, storage, recent job counts
GET  /api/fires?name=&year=      search CAL FIRE / WFIGS
POST /api/jobs {"fire_id": ...}  start (or reuse stored) mapping job
GET  /api/jobs/{job_id}          poll status / result
GET  /api/featured               fires with stored results (instant for visitors)
GET  /results/{dir}/{file}       PNG / GeoJSON layers (local file, or redirect to S3)
GET  /                           demo page

A run takes minutes (download + tiled inference), longer than a request should wait, so jobs
run in the background and the page polls.
  Local dev: `uvicorn burnmap.api:app`; jobs in memory, run in a thread, model loaded at startup.
  Lambda:    `burnmap.lambda_api.handler`; jobs in DynamoDB, run by the worker Lambda. This
             process never loads the model or imports torch.
"""

import logging
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from . import config, jobs as jobs_mod, metrics, perimeters, storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("rasterio").setLevel(logging.WARNING)
log = logging.getLogger("burnmap.api")

MODE = "lambda" if config.WORKER_FUNCTION else "local"

app = FastAPI(title="Burn Scar Mapper", version="0.3.0")
store = storage.make_store()
jobs = jobs_mod.make_jobs()
dispatcher = jobs_mod.make_dispatcher(jobs, store)
_model_error: str | None = None  # local mode only: the model failed to load at startup


@app.on_event("startup")
def _load_model():
    """Local dev only (the Lambda handler runs with lifespan off): fail loudly and early."""
    global _model_error
    try:
        from . import worker

        worker.get_model()
    except Exception as e:  # keep serving search + stored results without a model
        _model_error = f"{type(e).__name__}: {e}"
        log.warning("model not loaded: %s", _model_error)


def _checkpoint() -> str:
    return config.MODEL_PATH.name


def _reap(job: dict, now: float) -> dict:
    """Mark a job the worker never finished (killed at the time limit) as failed."""
    if jobs_mod.is_stale(job, now):
        err = f"timed out after {config.JOB_TIMEOUT_S}s"
        jobs.update(job["id"], status="error", error=err, finished=now)
        metrics.emit("JobFailed", job_id=job["id"], fire_id=job["fire_id"], error=err)
        job = {**job, "status": "error", "error": err, "finished": now}
    return job


@app.get("/api/health")
def health():
    now = time.time()
    counts: dict[str, int] = {}
    recent = [_reap(j, now) for j in jobs.recent(now - 3600)]
    for j in recent:
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    failures = sorted((j for j in recent if j["status"] == "error"), key=lambda j: j.get("finished", 0))
    return {"ok": True, "mode": MODE, "checkpoint": _checkpoint(), "threshold": config.THRESHOLD,
            "model_error": _model_error, "store": store.backend, "jobs_last_hour": counts,
            "new_jobs_last_hour": sum(not j.get("cached") for j in recent),
            "new_jobs_per_hour_limit": config.NEW_JOBS_PER_HOUR,
            "last_failure": ({k: failures[-1].get(k) for k in ("fire_id", "error", "finished")}
                             if failures else None)}


@app.get("/api/fires")
def fires(name: str, year: int | None = None, min_acres: float = 1000):
    if len(name.strip()) < 2:
        raise HTTPException(400, "name must be at least 2 characters")
    return [f.summary() for f in perimeters.search(name.strip(), year, min_acres)]


class JobRequest(BaseModel):
    fire_id: str


def _is_current(result: dict) -> bool:
    """A stored result is reusable if it came from the checkpoint and threshold being served.
    If the model can't run (local, failed to load), any stored result beats nothing."""
    if _model_error:
        return True
    m = result.get("model", {})
    return m.get("checkpoint") == _checkpoint() and abs(m.get("threshold", -1) - config.THRESHOLD) < 1e-9


@app.post("/api/jobs")
def create_job(req: JobRequest):
    if not storage.FIRE_ID_RE.match(req.fire_id):
        raise HTTPException(400, "fire_id must look like FRAP:<number> or WFIGS:<number>")
    now = time.time()
    job_id = uuid.uuid4().hex[:12]
    stored = store.get_result(storage.fire_dir(req.fire_id))
    if stored is not None and _is_current(stored):
        job = {"id": job_id, "fire_id": req.fire_id, "status": "done", "cached": True,
               "created": now, "finished": now, "result": stored}
        jobs.put(job)
        metrics.emit("CacheHit", fire_id=req.fire_id)
        return job
    if _model_error:
        raise HTTPException(503, f"Model not loaded ({_model_error}); only stored fires are available")

    recent = [_reap(j, now) for j in jobs.recent(now - max(3600, config.JOB_TIMEOUT_S + 120))]
    active = [j for j in recent if j["status"] in jobs_mod.ACTIVE]
    for j in active:  # don't run the same fire twice at once
        if j["fire_id"] == req.fire_id:
            return j
    # Limits are checked against a shared table without a lock: two simultaneous requests can
    # both squeeze in. That's acceptable for a demo; the worker's concurrency cap is the hard stop.
    if len(active) >= config.MAX_ACTIVE_JOBS:
        reason = f"{len(active)} jobs already in progress; try again in a few minutes"
    elif sum(1 for j in recent if not j.get("cached") and j["created"] >= now - 3600) >= config.NEW_JOBS_PER_HOUR:
        reason = "hourly limit for new fires reached; featured fires are still instant"
    else:
        reason = None
    if reason:
        metrics.emit("RateLimited", fire_id=req.fire_id)
        raise HTTPException(429, reason)

    job = {"id": job_id, "fire_id": req.fire_id, "status": "queued", "cached": False, "created": now}
    jobs.put(job)
    try:
        dispatcher.submit(job_id, req.fire_id)
    except Exception as e:
        jobs.update(job_id, status="error", error=f"could not start job: {e}", finished=time.time())
        log.exception("dispatch failed for %s", job_id)
        raise HTTPException(503, "could not start the job; try again shortly")
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if len(job_id) > 64:
        raise HTTPException(404, "unknown job")
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return _reap(job, time.time())


@app.get("/api/featured")
def featured():
    """Fires with stored results (instant for demo visitors)."""
    return [{**r["fire"], "model_f1": r["model"]["f1"], "model_acres": r["model"]["predicted_acres"]}
            for r in store.list_results()]


@app.get("/results/{fdir}/{name}")
def result_file(fdir: str, name: str):
    if not storage.FIRE_DIR_RE.match(fdir) or not storage.FILE_RE.match(name):
        raise HTTPException(404, "not found")
    local = store.local_file(fdir, name)
    if local is not None:
        return FileResponse(local)
    url = store.remote_url(fdir, name)
    if url is None:
        raise HTTPException(404, "not found")
    return RedirectResponse(url, status_code=307)


@app.get("/")
def index():
    return FileResponse(config.WEB_DIR / "index.html")
