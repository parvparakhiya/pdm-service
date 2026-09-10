"""Logging, metrics and the prediction audit trail.

The previous service had none of this: no logs, no metrics, no request ids, and
no record of what it predicted. That last one matters most. The model card
requires alert precision to be measured against confirmed work orders, and you
cannot do that if predictions were never written down. Without the audit log the
only question you can answer is "what did it score on a test set", never "is it
useful on the floor".

Zero third-party dependencies for logging: a small JSON formatter on stdlib
logging. Prometheus is optional -- if `prometheus_client` is absent the app
still runs and /metrics reports it clearly rather than pretending.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if fmt == "json"
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # uvicorn's own access log duplicates our middleware record
    logging.getLogger("uvicorn.access").disabled = True


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class _Metrics:
    """Prometheus if available; a no-op with an honest status otherwise."""

    def __init__(self) -> None:
        self.enabled = False
        self.reason = "prometheus_client not installed"
        try:
            from prometheus_client import Counter, Histogram, Gauge
            self.requests = Counter("pdm_requests_total", "Requests", ["endpoint", "status"])
            self.latency = Histogram("pdm_request_seconds", "Latency", ["endpoint"],
                                     buckets=(.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5))
            self.decisions = Counter("pdm_decisions_total", "Decisions",
                                     ["severity", "urgency"])
            self.mode_fired = Counter("pdm_mode_fired_total", "Spec predicate fired", ["mode"])
            self.hazard = Histogram("pdm_hazard_probability", "P(TWF within horizon)",
                                    buckets=(.01, .05, .1, .2, .4, .6, .8, 1.0))
            self.ready = Gauge("pdm_ready", "1 when the service can serve correct answers")
            self.enabled, self.reason = True, "ok"
        except ImportError:
            pass

    # graceful no-ops -----------------------------------------------------
    def observe_request(self, endpoint: str, status: int, seconds: float) -> None:
        if self.enabled:
            self.requests.labels(endpoint, str(status)).inc()
            self.latency.labels(endpoint).observe(seconds)

    def observe_decision(self, severity: str, urgency: str, modes: list[str],
                         hazard: float) -> None:
        if self.enabled:
            self.decisions.labels(severity, urgency).inc()
            for m in modes:
                self.mode_fired.labels(m).inc()
            self.hazard.observe(hazard)

    def set_ready(self, ready: bool) -> None:
        if self.enabled:
            self.ready.set(1 if ready else 0)

    def render(self) -> tuple[bytes, str]:
        if not self.enabled:
            raise RuntimeError(self.reason)
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return generate_latest(), CONTENT_TYPE_LATEST


METRICS = _Metrics()


# ---------------------------------------------------------------------------
# Prediction audit log
# ---------------------------------------------------------------------------
class PredictionAudit:
    def __init__(self, path: str, enabled: bool = True):
        self.enabled = enabled
        self.path = Path(path)
        self._lock = threading.Lock()
        if self.enabled:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logging.getLogger("pdm.audit").error(
                    "audit log disabled, cannot create %s: %s", self.path.parent, exc)
                self.enabled = False

    def record(self, payload: dict) -> None:
        if not self.enabled:
            return
        line = json.dumps(payload, default=str, separators=(",", ":"))
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            logging.getLogger("pdm.audit").error("audit write failed: %s", exc)


# ---------------------------------------------------------------------------
# In-process rate limiter
# ---------------------------------------------------------------------------
class TokenBucket:
    def __init__(self, per_minute: int):
        self.capacity = max(per_minute, 1)
        self.refill_per_sec = self.capacity / 60.0
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._state.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill_per_sec)
            if tokens < 1.0:
                self._state[key] = (tokens, now)
                return False
            self._state[key] = (tokens - 1.0, now)
            return True


def pid() -> int:
    return os.getpid()
