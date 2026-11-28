"""API, storage and metrics tests. Offline: the pipeline is stubbed and S3/CloudWatch are mocked with moto."""

import json
import time

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from burnmap import api, config, metrics, storage
from burnmap.model import LoadedModel


def _fake_result(fire_id, checkpoint="m.pt", threshold=0.5):
    return {"fire": {"id": fire_id, "name": "Test", "year": 2024},
            "model": {"checkpoint": checkpoint, "threshold": threshold, "f1": 0.8, "predicted_acres": 10.0},
            "layers": {}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "store", storage.LocalStore(tmp_path))
    monkeypatch.setattr(api, "_model", LoadedModel(net=None, channel_means=None, channel_stds=None, name="m.pt"))
    monkeypatch.setattr(api, "_jobs", {})
    monkeypatch.setattr(api, "_new_job_times", type(api._new_job_times)())
    monkeypatch.setattr(api.config, "THRESHOLD", 0.5)
    calls = []

    def fake_run(fire_id, model, out_root, threshold):
        calls.append(fire_id)
        d = out_root / storage.fire_dir(fire_id)
        d.mkdir(parents=True, exist_ok=True)
        r = _fake_result(fire_id, model.name, threshold)
        (d / "result.json").write_text(json.dumps(r))
        (d / "model_burn.png").write_bytes(b"png")
        return r

    monkeypatch.setattr(api.pipeline, "run", fake_run)
    c = TestClient(api.app)  # not a context manager: skips startup, so no real checkpoint load
    c.calls = calls
    return c


def _wait(client, job_id):
    for _ in range(100):
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_health_reports_model_and_limits(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["checkpoint"] == "m.pt" and h["threshold"] == 0.5 and h["store"] == "local"
    assert h["new_jobs_per_hour_limit"] == config.NEW_JOBS_PER_HOUR


@pytest.mark.parametrize("fid", ["FRAP", "FRAP:abc", "../etc:1", "S3:12", "FRAP:1/../2"])
def test_rejects_bad_fire_ids(client, fid):
    assert client.post("/api/jobs", json={"fire_id": fid}).status_code == 400


@pytest.mark.parametrize("path", ["/results/FRAP_1/..%2Fx.png", "/results/..%2F..%2Fsecret/x.png",
                                  "/results/FRAP_1/result.txt", "/results/FRAP_1/a.py"])
def test_results_route_rejects_odd_paths(client, path):
    assert client.get(path).status_code == 404


def test_job_runs_then_is_cached_then_rerun_when_threshold_changes(client, monkeypatch):
    j = _wait(client, client.post("/api/jobs", json={"fire_id": "FRAP:1"}).json()["id"])
    assert j["status"] == "done" and client.calls == ["FRAP:1"]
    assert client.get("/results/FRAP_1/model_burn.png").content == b"png"

    again = client.post("/api/jobs", json={"fire_id": "FRAP:1"}).json()
    assert again["cached"] and client.calls == ["FRAP:1"]

    monkeypatch.setattr(api.config, "THRESHOLD", 0.95)  # stored result is now stale
    j = _wait(client, client.post("/api/jobs", json={"fire_id": "FRAP:1"}).json()["id"])
    assert client.calls == ["FRAP:1", "FRAP:1"] and j["result"]["model"]["threshold"] == 0.95


def test_hourly_limit_returns_429_but_cached_fires_still_work(client, monkeypatch):
    monkeypatch.setattr(api.config, "NEW_JOBS_PER_HOUR", 1)
    _wait(client, client.post("/api/jobs", json={"fire_id": "FRAP:1"}).json()["id"])
    r = client.post("/api/jobs", json={"fire_id": "FRAP:2"})
    assert r.status_code == 429 and "hourly" in r.json()["detail"]
    assert client.post("/api/jobs", json={"fire_id": "FRAP:1"}).json()["cached"]


def test_failed_job_is_reported_in_health(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no clear scene")

    monkeypatch.setattr(api.pipeline, "run", boom)
    j = _wait(client, client.post("/api/jobs", json={"fire_id": "FRAP:9"}).json()["id"])
    assert j["status"] == "error" and "no clear scene" in j["error"]
    assert client.get("/api/health").json()["last_failure"]["fire_id"] == "FRAP:9"


@mock_aws
def test_s3_store_publish_read_list_and_redirect(tmp_path, monkeypatch):
    s3 = boto3.client("s3", region_name="us-west-2")
    s3.create_bucket(Bucket="bsm-test", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
    st = storage.S3Store("bsm-test", "results/", root=tmp_path / "work", client=s3)
    d = st.work_dir() / "FRAP_7"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(_fake_result("FRAP:7")))
    (d / "perimeter.geojson").write_text("{}")
    st.publish("FRAP_7")
    head = s3.head_object(Bucket="bsm-test", Key="results/FRAP_7/perimeter.geojson")
    assert head["ContentType"] == "application/geo+json"

    fresh = storage.S3Store("bsm-test", "results/", root=tmp_path / "empty", client=s3)  # e.g. after a redeploy
    assert fresh.get_result("FRAP_7")["fire"]["id"] == "FRAP:7"
    assert fresh.get_result("FRAP_8") is None
    assert [r["fire"]["id"] for r in fresh.list_results()] == ["FRAP:7"]

    monkeypatch.setattr(api, "store", fresh)
    r = TestClient(api.app).get("/results/FRAP_7/perimeter.geojson", follow_redirects=False)
    assert r.status_code == 307 and "FRAP_7/perimeter.geojson" in r.headers["location"]


@mock_aws
def test_metrics_go_to_cloudwatch_and_never_raise(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    monkeypatch.setattr(metrics.config, "METRICS_BACKEND", "cloudwatch")
    monkeypatch.setattr(metrics, "_cw", None)
    metrics.emit("JobFailed", fire_id="FRAP:1")
    names = [m["MetricName"] for m in boto3.client("cloudwatch", region_name="us-west-2").list_metrics(
        Namespace=config.METRICS_NAMESPACE)["Metrics"]]
    assert names == ["JobFailed"]

    monkeypatch.setattr(metrics, "_cw", object())  # broken client
    metrics.emit("JobFailed")  # logs a warning, doesn't raise
