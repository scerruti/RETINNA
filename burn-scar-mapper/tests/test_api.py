"""API, jobs, worker, storage and metrics tests. Offline: the pipeline is stubbed; S3, DynamoDB
and CloudWatch are mocked with moto."""

import json
import time

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from burnmap import api, config, jobs as jobs_mod, metrics, storage, worker
from burnmap.model import LoadedModel

CKPT = config.MODEL_PATH.name


def _fake_result(fire_id, checkpoint=CKPT, threshold=0.5):
    return {"fire": {"id": fire_id, "name": "Test", "year": 2024},
            "model": {"checkpoint": checkpoint, "threshold": threshold, "f1": 0.8, "predicted_acres": 10.5},
            "layers": {}}


def _fake_pipeline(calls):
    def run(fire_id, model, out_root, threshold):
        calls.append(fire_id)
        d = out_root / storage.fire_dir(fire_id)
        d.mkdir(parents=True, exist_ok=True)
        r = _fake_result(fire_id, CKPT, threshold)
        (d / "result.json").write_text(json.dumps(r))
        (d / "model_burn.png").write_bytes(b"png")
        return r
    return run


@pytest.fixture
def client(tmp_path, monkeypatch):
    st = storage.LocalStore(tmp_path)
    jb = jobs_mod.MemoryJobs()
    monkeypatch.setattr(api, "store", st)
    monkeypatch.setattr(api, "jobs", jb)
    monkeypatch.setattr(api, "dispatcher", jobs_mod.ThreadDispatcher(jb, st))
    monkeypatch.setattr(api, "_model_error", None)
    monkeypatch.setattr(api.config, "THRESHOLD", 0.5)
    monkeypatch.setattr(worker, "_model", LoadedModel(net=None, channel_means=None, channel_stds=None, name=CKPT))
    calls = []
    monkeypatch.setattr(worker.pipeline, "run", _fake_pipeline(calls))
    c = TestClient(api.app)  # not a context manager: skips startup, so no real checkpoint load
    c.calls = calls
    return c


def _wait(client, job_id):
    for _ in range(200):
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_health_reports_mode_checkpoint_and_limits(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["mode"] == "local" and h["checkpoint"] == CKPT and h["threshold"] == 0.5
    assert h["store"] == "local" and h["new_jobs_per_hour_limit"] == config.NEW_JOBS_PER_HOUR


@pytest.mark.parametrize("fid", ["FRAP", "FRAP:abc", "../etc:1", "S3:12", "FRAP:1/../2"])
def test_rejects_bad_fire_ids(client, fid):
    assert client.post("/api/jobs", json={"fire_id": fid}).status_code == 400


@pytest.mark.parametrize("path", ["/results/FRAP_1/..%2Fx.png", "/results/..%2F..%2Fsecret/x.png",
                                  "/results/FRAP_1/result.txt", "/results/FRAP_1/a.py"])
def test_results_route_rejects_odd_paths(client, path):
    assert client.get(path).status_code == 404


def test_index_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()


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


def test_active_limit_and_duplicate_fire(client, monkeypatch):
    monkeypatch.setattr(api, "dispatcher", type("NoRun", (), {"submit": lambda *a: None})())
    monkeypatch.setattr(api.config, "MAX_ACTIVE_JOBS", 2)
    a = client.post("/api/jobs", json={"fire_id": "FRAP:1"}).json()
    assert client.post("/api/jobs", json={"fire_id": "FRAP:1"}).json()["id"] == a["id"]  # same fire: same job
    client.post("/api/jobs", json={"fire_id": "FRAP:2"})
    r = client.post("/api/jobs", json={"fire_id": "FRAP:3"})
    assert r.status_code == 429 and "in progress" in r.json()["detail"]


def test_job_killed_at_time_limit_is_reported_as_timed_out(client, monkeypatch):
    monkeypatch.setattr(api, "dispatcher", type("NoRun", (), {"submit": lambda *a: None})())
    j = client.post("/api/jobs", json={"fire_id": "FRAP:4"}).json()
    api.jobs.update(j["id"], created=time.time() - config.JOB_TIMEOUT_S - 600)
    got = client.get(f"/api/jobs/{j['id']}").json()
    assert got["status"] == "error" and "timed out" in got["error"]


def test_failed_job_is_reported_in_health(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no clear scene")

    monkeypatch.setattr(worker.pipeline, "run", boom)
    j = _wait(client, client.post("/api/jobs", json={"fire_id": "FRAP:9"}).json()["id"])
    assert j["status"] == "error" and "no clear scene" in j["error"]
    assert client.get("/api/health").json()["last_failure"]["fire_id"] == "FRAP:9"


def test_dispatch_failure_returns_503_and_marks_job(client, monkeypatch):
    def fail(*a):
        raise RuntimeError("throttled")

    monkeypatch.setattr(api, "dispatcher", type("Fail", (), {"submit": lambda self, *a: fail()})())
    r = client.post("/api/jobs", json={"fire_id": "FRAP:5"})
    assert r.status_code == 503
    assert [j["status"] for j in api.jobs.recent(0)] == ["error"]


# --- AWS pieces (moto) ---

@pytest.fixture
def aws(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    with mock_aws():
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket="bsm-test", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
        ddb = boto3.resource("dynamodb")
        ddb.create_table(TableName="bsm-jobs", KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
                         AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
                         BillingMode="PAY_PER_REQUEST")
        yield s3, ddb


def test_s3_store_publish_read_list_and_redirect(aws, tmp_path, monkeypatch):
    s3, _ = aws
    st = storage.S3Store("bsm-test", "results/", root=tmp_path / "work", client=s3)
    d = st.work_dir() / "FRAP_7"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(_fake_result("FRAP:7")))
    (d / "perimeter.geojson").write_text("{}")
    st.publish("FRAP_7")
    head = s3.head_object(Bucket="bsm-test", Key="results/FRAP_7/perimeter.geojson")
    assert head["ContentType"] == "application/geo+json"

    fresh = storage.S3Store("bsm-test", "results/", root=tmp_path / "empty", client=s3)  # another Lambda
    assert fresh.get_result("FRAP_7")["fire"]["id"] == "FRAP:7"
    assert fresh.get_result("FRAP_8") is None
    assert [r["fire"]["id"] for r in fresh.list_results()] == ["FRAP:7"]

    monkeypatch.setattr(api, "store", fresh)
    r = TestClient(api.app).get("/results/FRAP_7/perimeter.geojson", follow_redirects=False)
    assert r.status_code == 307 and "FRAP_7/perimeter.geojson" in r.headers["location"]


def test_dynamo_jobs_roundtrip_floats_and_recent(aws):
    _, ddb = aws
    jb = jobs_mod.DynamoJobs("bsm-jobs", resource=ddb)
    now = time.time()
    jb.put({"id": "a", "fire_id": "FRAP:1", "status": "queued", "cached": False, "created": now})
    jb.put({"id": "old", "fire_id": "FRAP:2", "status": "done", "cached": False, "created": now - 7200})
    jb.update("a", status="done", result=_fake_result("FRAP:1"), finished=now + 1)
    a = jb.get("a")
    assert a["status"] == "done" and a["result"]["model"]["predicted_acres"] == 10.5 and isinstance(a["created"], int)
    assert jb.get("missing") is None
    assert [j["id"] for j in jb.recent(now - 3600)] == ["a"]


def test_worker_lambda_handler_end_to_end(aws, tmp_path, monkeypatch):
    s3, ddb = aws
    st = storage.S3Store("bsm-test", "results/", root=tmp_path / "w", client=s3)
    jb = jobs_mod.DynamoJobs("bsm-jobs", resource=ddb)
    monkeypatch.setattr(worker.storage, "make_store", lambda: st)
    monkeypatch.setattr(worker.jobs_mod, "make_jobs", lambda: jb)
    monkeypatch.setattr(worker, "_model", LoadedModel(net=None, channel_means=None, channel_stds=None, name=CKPT))
    calls = []
    monkeypatch.setattr(worker.pipeline, "run", _fake_pipeline(calls))
    jb.put({"id": "j1", "fire_id": "FRAP:3", "status": "queued", "cached": False, "created": time.time()})

    worker.handler({"job_id": "j1", "fire_id": "FRAP:3"}, None)

    j = jb.get("j1")
    assert j["status"] == "done" and j["result"]["fire"]["id"] == "FRAP:3"
    assert s3.head_object(Bucket="bsm-test", Key="results/FRAP_3/result.json")
    assert not (tmp_path / "w" / "FRAP_3").exists()  # /tmp cleaned after publishing


def test_lambda_dispatcher_invokes_worker_async():
    sent = {}

    class FakeLambda:
        def invoke(self, **kw):
            sent.update(kw)

    jobs_mod.LambdaDispatcher("bsm-worker", client=FakeLambda()).submit("j9", "FRAP:9")
    assert sent["FunctionName"] == "bsm-worker" and sent["InvocationType"] == "Event"
    assert json.loads(sent["Payload"]) == {"job_id": "j9", "fire_id": "FRAP:9"}


def test_api_lambda_handler_serves_function_url_event(client):
    from burnmap import lambda_api

    event = {"version": "2.0", "routeKey": "$default", "rawPath": "/api/health", "rawQueryString": "",
             "headers": {"host": "abc.lambda-url.us-west-2.on.aws"},
             "requestContext": {"http": {"method": "GET", "path": "/api/health", "sourceIp": "1.2.3.4",
                                         "protocol": "HTTP/1.1"}, "domainName": "abc.lambda-url.us-west-2.on.aws",
                                "stage": "$default"},
             "isBase64Encoded": False}
    r = lambda_api.handler(event, type("Ctx", (), {"aws_request_id": "x"})())
    assert r["statusCode"] == 200 and json.loads(r["body"])["checkpoint"] == CKPT


def test_metrics_go_to_cloudwatch_and_never_raise(aws, monkeypatch):
    monkeypatch.setattr(metrics.config, "METRICS_BACKEND", "cloudwatch")
    monkeypatch.setattr(metrics, "_cw", None)
    metrics.emit("JobFailed", fire_id="FRAP:1")
    names = [m["MetricName"] for m in boto3.client("cloudwatch").list_metrics(
        Namespace=config.METRICS_NAMESPACE)["Metrics"]]
    assert names == ["JobFailed"]

    monkeypatch.setattr(metrics, "_cw", object())  # broken client
    metrics.emit("JobFailed")  # logs a warning, doesn't raise
