"""Job metrics: one JSON log line per event, and optionally CloudWatch custom metrics.

Metrics (namespace config.METRICS_NAMESPACE):
  JobSucceeded / JobFailed (Count), JobDuration (Seconds), CacheHit (Count), RateLimited (Count)
The CloudFormation template alarms on JobFailed and charts all of them on a dashboard.
"""

import json
import logging
import time

from . import config

log = logging.getLogger("burnmap.metrics")
_cw = None


def _client():
    global _cw
    if _cw is None:
        import boto3

        _cw = boto3.client("cloudwatch")
    return _cw


def emit(name: str, value: float = 1.0, unit: str = "Count", **fields) -> None:
    """Never raises: monitoring must not break a job."""
    log.info(json.dumps({"metric": name, "value": value, "unit": unit, "ts": round(time.time(), 3), **fields}))
    if config.METRICS_BACKEND != "cloudwatch":
        return
    try:
        _client().put_metric_data(Namespace=config.METRICS_NAMESPACE,
                                  MetricData=[{"MetricName": name, "Value": value, "Unit": unit}])
    except Exception as e:
        log.warning("put_metric_data failed for %s: %s", name, e)
