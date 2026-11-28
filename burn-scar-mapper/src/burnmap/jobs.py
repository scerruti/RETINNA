"""Job records and how jobs get run.

Local dev: jobs live in memory and run in a background thread of the API process.
Lambda: jobs live in DynamoDB (shared by every API and worker invocation, expired by TTL),
and the API starts the worker Lambda asynchronously.

A job record: id, fire_id, status (queued | running | done | error), cached, created,
started, finished, result (dict), error. Times are epoch seconds.
"""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import config

log = logging.getLogger(__name__)
ACTIVE = ("queued", "running")
TTL_S = 7 * 24 * 3600


def is_stale(job: dict, now: float) -> bool:
    """An active job older than the worker's hard limit was killed (Lambda timeout, crash)."""
    return job["status"] in ACTIVE and now - job["created"] > config.JOB_TIMEOUT_S + 120


class MemoryJobs:
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def put(self, job: dict) -> None:
        with self._lock:
            self._jobs[job["id"]] = dict(job)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            j = self._jobs.get(job_id)
            return dict(j) if j else None

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            self._jobs[job_id].update(fields)

    def recent(self, since: float) -> list[dict]:
        """Jobs created since `since` (used for active-job and hourly limits)."""
        with self._lock:
            for jid in [k for k, v in self._jobs.items() if v["created"] < time.time() - TTL_S]:
                del self._jobs[jid]
            return [dict(j) for j in self._jobs.values() if j["created"] >= since]


class DynamoJobs:
    """Table with partition key `id` (S) and TTL on `ttl`. Floats can't go into DynamoDB
    through boto3 without Decimal, so the result dict is stored as a JSON string."""

    def __init__(self, table: str, resource=None):
        import boto3

        self.table = (resource or boto3.resource("dynamodb")).Table(table)

    @staticmethod
    def _out(item: dict) -> dict:
        j = {k: (int(v) if k in ("created", "started", "finished", "ttl") else v) for k, v in item.items()}
        if "result" in j:
            j["result"] = json.loads(j["result"])
        j.pop("ttl", None)
        return j

    @staticmethod
    def _in(fields: dict) -> dict:
        f = {k: (int(v) if k in ("created", "started", "finished") else v) for k, v in fields.items()}
        if "result" in f:
            f["result"] = json.dumps(f["result"])
        return f

    def put(self, job: dict) -> None:
        self.table.put_item(Item={**self._in(job), "ttl": int(job["created"]) + TTL_S})

    def get(self, job_id: str) -> dict | None:
        item = self.table.get_item(Key={"id": job_id}).get("Item")
        return self._out(item) if item else None

    def update(self, job_id: str, **fields) -> None:
        f = self._in(fields)
        names = {f"#{k}": k for k in f}
        values = {f":{k}": v for k, v in f.items()}
        self.table.update_item(Key={"id": job_id}, UpdateExpression="SET " + ", ".join(f"#{k} = :{k}" for k in f),
                               ExpressionAttributeNames=names, ExpressionAttributeValues=values)

    def recent(self, since: float) -> list[dict]:
        # A demo sees a handful of jobs per hour; a filtered scan is simpler than an index.
        from boto3.dynamodb.conditions import Attr

        out, kw = [], {"FilterExpression": Attr("created").gte(int(since))}
        while True:
            page = self.table.scan(**kw)
            out += [self._out(i) for i in page.get("Items", [])]
            if "LastEvaluatedKey" not in page:
                return out
            kw["ExclusiveStartKey"] = page["LastEvaluatedKey"]


class ThreadDispatcher:
    """Local dev: run jobs one at a time in this process."""

    def __init__(self, jobs, store):
        self.jobs, self.store = jobs, store
        self._executor = ThreadPoolExecutor(max_workers=1)

    def submit(self, job_id: str, fire_id: str) -> None:
        from . import worker  # imports torch; only needed where jobs actually run

        self._executor.submit(worker.run_job, job_id, fire_id, self.jobs, self.store)


class LambdaDispatcher:
    """Deployed: fire-and-forget invoke of the worker function."""

    def __init__(self, function: str, client=None):
        import boto3

        self.function = function
        self.client = client or boto3.client("lambda")

    def submit(self, job_id: str, fire_id: str) -> None:
        self.client.invoke(FunctionName=self.function, InvocationType="Event",
                           Payload=json.dumps({"job_id": job_id, "fire_id": fire_id}).encode())


def make_jobs():
    return DynamoJobs(config.JOBS_TABLE) if config.JOBS_TABLE else MemoryJobs()


def make_dispatcher(jobs, store):
    return LambdaDispatcher(config.WORKER_FUNCTION) if config.WORKER_FUNCTION else ThreadDispatcher(jobs, store)
