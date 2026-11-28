"""Where finished results live.

Local mode (dev): RESULTS_DIR/<fire_dir>/{result.json, layers}.
S3 mode (deployed): the same files are published to s3://BUCKET/PREFIX<fire_dir>/ when a job
finishes, so results survive container restarts and redeploys (App Runner disk is ephemeral).
Layers are served by redirecting to short-lived presigned URLs; the bucket stays private.
"""

import json
import logging
import mimetypes
import re
import threading
import time
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

FIRE_ID_RE = re.compile(r"^(FRAP|WFIGS):\d{1,12}$")   # OBJECTIDs from perimeters.py
FIRE_DIR_RE = re.compile(r"^(FRAP|WFIGS)_\d{1,12}$")
FILE_RE = re.compile(r"^[a-z_]{1,40}\.(png|jpg|geojson|json)$")


def fire_dir(fire_id: str) -> str:
    return fire_id.replace(":", "_")


class LocalStore:
    backend = "local"

    def __init__(self, root: Path = config.RESULTS_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def work_dir(self) -> Path:
        return self.root

    def get_result(self, fdir: str) -> dict | None:
        p = self.root / fdir / "result.json"
        return json.loads(p.read_text()) if p.exists() else None

    def publish(self, fdir: str) -> None:
        pass  # already on disk

    def local_file(self, fdir: str, name: str) -> Path | None:
        p = self.root / fdir / name
        return p if p.is_file() else None

    def remote_url(self, fdir: str, name: str) -> str | None:
        return None

    def list_results(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted(self.root.glob("*/result.json"))]


class S3Store(LocalStore):
    backend = "s3"
    LIST_TTL_S = 300

    def __init__(self, bucket: str, prefix: str = config.RESULTS_PREFIX, root: Path = config.RESULTS_DIR, client=None):
        super().__init__(root)
        import boto3

        self.s3 = client or boto3.client("s3")
        self.bucket, self.prefix = bucket, prefix
        self._list_cache: tuple[float, list[dict]] | None = None
        self._lock = threading.Lock()

    def _key(self, fdir: str, name: str) -> str:
        return f"{self.prefix}{fdir}/{name}"

    def get_result(self, fdir: str) -> dict | None:
        local = super().get_result(fdir)
        if local is not None:
            return local
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=self._key(fdir, "result.json"))["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None
        return json.loads(body)

    def publish(self, fdir: str) -> None:
        src = self.root / fdir
        files = sorted(p for p in src.iterdir() if p.is_file())
        # result.json last: its presence is what marks a result as complete
        files.sort(key=lambda p: p.name == "result.json")
        for p in files:
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            if p.suffix == ".geojson":
                ctype = "application/geo+json"
            self.s3.upload_file(str(p), self.bucket, self._key(fdir, p.name), ExtraArgs={"ContentType": ctype})
        with self._lock:
            self._list_cache = None
        log.info("published %d files to s3://%s/%s%s/", len(files), self.bucket, self.prefix, fdir)

    def remote_url(self, fdir: str, name: str) -> str | None:
        return self.s3.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": self._key(fdir, name)}, ExpiresIn=3600)

    def list_results(self) -> list[dict]:
        with self._lock:
            if self._list_cache and time.time() - self._list_cache[0] < self.LIST_TTL_S:
                return self._list_cache[1]
        out = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("/result.json"):
                    out.append(json.loads(self.s3.get_object(Bucket=self.bucket, Key=obj["Key"])["Body"].read()))
        with self._lock:
            self._list_cache = (time.time(), out)
        return out


def make_store():
    if config.RESULTS_BUCKET:
        return S3Store(config.RESULTS_BUCKET)
    return LocalStore()
