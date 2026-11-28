"""FastAPI service.

GET  /api/health                 model, threshold, storage, job counts (App Runner health check)
GET  /api/fires?name=&year=      search CAL FIRE / WFIGS
POST /api/jobs {"fire_id": ...}  start (or reuse cached) mapping job
GET  /api/jobs/{job_id}          poll status / result
GET  /api/featured               fires with stored results (instant for visitors)
GET  /results/{dir}/{file}       PNG / GeoJSON layers (local file, or redirect to S3)
GET  /                           demo page

A run takes ~1-5 min on CPU (download + tiled inference), longer than most proxies allow
for one request, so jobs run in a background worker and the page polls.

The job table is in memory, so the service must run as ONE instance (the CloudFormation
template pins App Runner to MinSize=MaxSize=1). Results themselves persist in S3.
"""

import logging
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from . import config, metrics, perimeters, pipeline, storage
from .model import LoadedModel, load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("rasterio").setLevel(logging.WARNING)
log = logging.getLogger("burnmap.api")

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
JOB_TTL_S = 3600  # finished jobs are dropped from memory after an hour; results stay in the store

app = FastAPI(title="Burn Scar Mapper", version="0.2.0")
store = storage.make_store()
_executor = ThreadPoolExecutor(max_workers=1)  # one CPU-heavy job at a time
_jobs: dict[str, dict] = {}
_new_job_times: deque[float] = deque()  # start times of uncached jobs, for the hourly cap
_lock = threading.Lock()
_model: LoadedModel | None = None
_model_error: str | None = None
_started = time.time()
_last_failure: dict | None = None


@app.on_event("startup")
def _load_model():
    global _model, _model_error
    try:
        _model = load_checkpoint(config.MODEL_PATH)
        log.info("model %s loaded; threshold %.3f; store %s", _model.name, config.THRESHOLD, store.backend)
    except Exception as e:  # keep serving search + stored results without a model
        _model_error = f"{type(e).__name__}: {e}"
        log.warning("model not loaded: %s", _model_error)


def _prune_jobs(now: float) -> None:
    for jid in [j for j, v in _jobs.items() if v["status"] in ("done", "error") and now - v.get("finished", now) > JOB_TTL_S]:
        del _jobs[jid]
    while _new_job_times and now - _new_job_times[0] > 3600:
        _new_job_times.popleft()


@app.get("/api/health")
def health():
    with _lock:
        counts: dict[str, int] = {}
        for j in _jobs.values():
            counts[j["status"]] = counts.get(j["status"], 0) + 1
        new_last_hour = len(_new_job_times)
    return {"ok": True, "uptime_s": round(time.time() - _started),
            "model_loaded": _model is not None, "model_error": _model_error,
            "checkpoint": _model.name if _model else None, "threshold": config.THRESHOLD,
            "model_metrics": _model.metrics if _model else None,
            "store": store.backend, "jobs": counts,
            "new_jobs_last_hour": new_last_hour, "new_jobs_per_hour_limit": config.NEW_JOBS_PER_HOUR,
            "last_failure": _last_failure}


@app.get("/api/fires")
def fires(name: str, year: int | None = None, min_acres: float = 1000):
    if len(name.strip()) < 2:
        raise HTTPException(400, "name must be at least 2 characters")
    return [f.summary() for f in perimeters.search(name.strip(), year, min_acres)]


class JobRequest(BaseModel):
    fire_id: str


def _is_current(result: dict) -> bool:
    """A stored result is reusable if it came from the model and threshold now being served.
    With no model loaded, any stored result is better than nothing."""
    if _model is None:
        return True
    m = result.get("model", {})
    return m.get("checkpoint") == _model.name and abs(m.get("threshold", -1) - config.THRESHOLD) < 1e-9


def _run(job_id: str, fire_id: str):
    global _last_failure
    t0 = time.time()
    with _lock:
        _jobs[job_id]["status"] = "running"
    try:
        result = pipeline.run(fire_id, _model, store.work_dir(), config.THRESHOLD)
        store.publish(storage.fire_dir(fire_id))
        with _lock:
            _jobs[job_id].update(status="done", result=result, finished=time.time())
        metrics.emit("JobSucceeded", job_id=job_id, fire_id=fire_id)
        metrics.emit("JobDuration", round(time.time() - t0, 1), "Seconds", job_id=job_id, fire_id=fire_id)
    except Exception as e:
        log.exception("job %s failed", job_id)
        err = f"{type(e).__name__}: {e}"
        with _lock:
            _jobs[job_id].update(status="error", error=err, finished=time.time())
            _last_failure = {"fire_id": fire_id, "error": err, "at": round(time.time())}
        metrics.emit("JobFailed", job_id=job_id, fire_id=fire_id, error=err)


@app.post("/api/jobs")
def create_job(req: JobRequest):
    if not storage.FIRE_ID_RE.match(req.fire_id):
        raise HTTPException(400, "fire_id must look like FRAP:<number> or WFIGS:<number>")
    job_id = uuid.uuid4().hex[:12]
    cached = store.get_result(storage.fire_dir(req.fire_id))
    if cached is not None and _is_current(cached):
        job = {"id": job_id, "fire_id": req.fire_id, "status": "done", "cached": True,
               "result": cached, "finished": time.time()}
        with _lock:
            _jobs[job_id] = job
        metrics.emit("CacheHit", fire_id=req.fire_id)
        return job
    if _model is None:
        raise HTTPException(503, f"Model not loaded ({_model_error}); only stored fires are available")
    now = time.time()
    with _lock:
        _prune_jobs(now)
        active = [j for j in _jobs.values() if j["status"] in ("queued", "running")]
        for j in active:  # don't queue the same fire twice
            if j["fire_id"] == req.fire_id:
                return j
        reason = None
        if len(active) >= config.MAX_ACTIVE_JOBS:
            reason = f"{len(active)} jobs already in progress; try again in a few minutes"
        elif len(_new_job_times) >= config.NEW_JOBS_PER_HOUR:
            reason = "hourly limit for new fires reached; featured fires are still instant"
        if reason is None:
            _new_job_times.append(now)
            _jobs[job_id] = {"id": job_id, "fire_id": req.fire_id, "status": "queued"}
    if reason:
        metrics.emit("RateLimited", fire_id=req.fire_id)
        raise HTTPException(429, reason)
    _executor.submit(_run, job_id, req.fire_id)
    return _jobs[job_id]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return job


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
    return FileResponse(WEB_DIR / "index.html")
