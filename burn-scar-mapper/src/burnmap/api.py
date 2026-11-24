"""FastAPI service.

GET  /api/health                 model status
GET  /api/fires?name=&year=      search CAL FIRE / WFIGS
POST /api/jobs {"fire_id": ...}  start (or reuse cached) mapping job
GET  /api/jobs/{job_id}          poll status / result
GET  /results/{dir}/{file}       PNG / GeoJSON layers
GET  /                           demo page

A run takes ~1-5 min on CPU (download + tiled inference), longer than most proxies allow
for one request, so jobs run in a background worker and the page polls.
"""

import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, perimeters, pipeline
from .model import LoadedModel, load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("rasterio").setLevel(logging.WARNING)
log = logging.getLogger("burnmap.api")

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Burn Scar Mapper", version="0.1.0")
_executor = ThreadPoolExecutor(max_workers=1)  # one CPU-heavy job at a time
_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_model: LoadedModel | None = None
_model_error: str | None = None


@app.on_event("startup")
def _load_model():
    global _model, _model_error
    try:
        _model = load_checkpoint(config.MODEL_PATH)
        log.info("model loaded from %s", config.MODEL_PATH)
    except Exception as e:  # keep serving search + cached results without a model
        _model_error = f"{type(e).__name__}: {e}"
        log.warning("model not loaded: %s", _model_error)


@app.get("/api/health")
def health():
    return {"ok": True, "model_loaded": _model is not None, "model_error": _model_error,
            "model_metrics": _model.metrics if _model else None}


@app.get("/api/fires")
def fires(name: str, year: int | None = None, min_acres: float = 1000):
    if len(name.strip()) < 2:
        raise HTTPException(400, "name must be at least 2 characters")
    return [f.summary() for f in perimeters.search(name.strip(), year, min_acres)]


class JobRequest(BaseModel):
    fire_id: str


def _result_dir(fire_id: str) -> Path:
    return config.RESULTS_DIR / fire_id.replace(":", "_")


def _run(job_id: str, fire_id: str):
    with _lock:
        _jobs[job_id]["status"] = "running"
    try:
        result = pipeline.run(fire_id, _model)
        with _lock:
            _jobs[job_id].update(status="done", result=result)
    except Exception as e:
        log.exception("job %s failed", job_id)
        with _lock:
            _jobs[job_id].update(status="error", error=f"{type(e).__name__}: {e}")


@app.post("/api/jobs")
def create_job(req: JobRequest):
    if not req.fire_id.split(":")[0] in ("FRAP", "WFIGS"):
        raise HTTPException(400, "fire_id must look like FRAP:<id> or WFIGS:<id>")
    cached = _result_dir(req.fire_id) / "result.json"
    job_id = uuid.uuid4().hex[:12]
    if cached.exists():
        job = {"id": job_id, "fire_id": req.fire_id, "status": "done", "cached": True,
               "result": json.loads(cached.read_text())}
        with _lock:
            _jobs[job_id] = job
        return job
    if _model is None:
        raise HTTPException(503, f"Model not loaded ({_model_error}); only cached fires are available")
    with _lock:
        for j in _jobs.values():  # don't queue the same fire twice
            if j["fire_id"] == req.fire_id and j["status"] in ("queued", "running"):
                return j
        _jobs[job_id] = {"id": job_id, "fire_id": req.fire_id, "status": "queued"}
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
    """Fires already computed and cached on disk (instant for demo visitors)."""
    out = []
    for p in sorted(config.RESULTS_DIR.glob("*/result.json")):
        r = json.loads(p.read_text())
        out.append({**r["fire"], "model_f1": r["model"]["f1"], "model_acres": r["model"]["predicted_acres"]})
    return out


app.mount("/results", StaticFiles(directory=config.RESULTS_DIR), name="results")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")
