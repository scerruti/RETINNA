"""Runs one mapping job. Called from a thread in local dev, or as the worker Lambda's handler.

Lambda: the API invokes this function asynchronously with {"job_id", "fire_id"}. Retries are
off (a failed run is reported, not repeated), and the function's timeout is config.JOB_TIMEOUT_S.
"""

import logging
import os
import shutil
import time

from . import config, jobs as jobs_mod, metrics, pipeline, storage
from .model import LoadedModel, load_checkpoint

log = logging.getLogger("burnmap.worker")
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("rasterio").setLevel(logging.WARNING)

_model: LoadedModel | None = None


def get_model() -> LoadedModel:
    """Loaded once per process (one Lambda cold start), then reused by warm invocations."""
    global _model
    if _model is None:
        import torch

        torch.set_num_threads(os.cpu_count() or 1)
        _model = load_checkpoint(config.MODEL_PATH)
        log.info("model %s loaded (%d threads)", _model.name, torch.get_num_threads())
    return _model


def run_job(job_id: str, fire_id: str, jobs, store, cleanup: bool = False) -> None:
    t0 = time.time()
    jobs.update(job_id, status="running", started=t0)
    fdir = storage.fire_dir(fire_id)
    try:
        result = pipeline.run(fire_id, get_model(), store.work_dir(), config.THRESHOLD)
        store.publish(fdir)
        jobs.update(job_id, status="done", result=result, finished=time.time())
        metrics.emit("JobSucceeded", job_id=job_id, fire_id=fire_id)
        metrics.emit("JobDuration", round(time.time() - t0, 1), "Seconds", job_id=job_id, fire_id=fire_id)
    except Exception as e:
        log.exception("job %s failed", job_id)
        jobs.update(job_id, status="error", error=f"{type(e).__name__}: {e}", finished=time.time())
        metrics.emit("JobFailed", job_id=job_id, fire_id=fire_id, error=f"{type(e).__name__}: {e}")
    finally:
        if cleanup:  # Lambda /tmp persists across warm invocations; results are already in S3
            shutil.rmtree(store.work_dir() / fdir, ignore_errors=True)


def handler(event, context):
    run_job(event["job_id"], event["fire_id"], jobs_mod.make_jobs(), storage.make_store(), cleanup=True)
    return {"job_id": event["job_id"]}
