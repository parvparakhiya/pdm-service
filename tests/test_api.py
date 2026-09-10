"""HTTP-layer tests. Requires FastAPI; skipped automatically when absent.

The scoring logic is covered in test_service.py without a web server. What is
tested here is only what the HTTP layer adds: routing, auth, rate limiting,
status codes, and the guarantee that error responses never leak internals.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi.testclient import TestClient                       # noqa: E402

os.environ["PDM_ENVIRONMENT"] = "local"
os.environ["PDM_API_KEYS"] = "test-key"
os.environ["PDM_PREDICTION_LOG_ENABLED"] = "false"

from app.config import get_settings                             # noqa: E402
from app.main import app                                        # noqa: E402

get_settings.cache_clear()

PAYLOAD = {"machine_id": "CNC-014", "product_type": "M", "temp_air_k": 298.2,
           "temp_process_k": 308.7, "speed_rpm": 1408.0, "torque_nm": 46.3,
           "tool_wear_min": 115.0}
AUTH = {"x-api-key": "test-key"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --- liveness vs readiness are different questions -------------------------
def test_health_is_liveness_only(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_ready_reports_model_state(client):
    r = client.get("/ready")
    body = r.json()
    assert r.status_code == 200
    assert body["ready"] is True
    assert body["rule_engine"] is True and body["hazard_model"] is True
    assert body["onnx_classifier"] in ("loaded", "disabled")
    assert len(body["contract_digest"]) == 16


# --- auth ------------------------------------------------------------------
def test_scoring_requires_a_key(client):
    r = client.post("/api/v1/score", json=PAYLOAD)
    assert r.status_code == 401
    assert r.json()["code"] == "unauthorized"


def test_wrong_key_is_rejected(client):
    r = client.post("/api/v1/score", json=PAYLOAD, headers={"x-api-key": "nope"})
    assert r.status_code == 401


# --- happy path ------------------------------------------------------------
def test_score_returns_the_full_contract(client):
    r = client.post("/api/v1/score", json=PAYLOAD, headers=AUTH)
    assert r.status_code == 200
    b = r.json()
    assert {f["mode"] for f in b["findings"]} == {"TWF", "HDF", "PWF", "OSF"}
    assert b["decision"]["severity"] in ("NOMINAL", "WATCH", "DEGRADED", "CRITICAL")
    assert b["tool_life"]["estimator"] == "analytic_uniform_tool_life"
    assert b["machine_id"] == "CNC-014"
    assert len(b["policy_fingerprint"]) == 16
    assert "predicted_rul_minutes" not in b       # the deleted model must stay deleted


def test_request_id_is_echoed(client):
    r = client.post("/api/v1/score", json=PAYLOAD, headers={**AUTH, "x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"
    assert r.json()["request_id"] == "abc123"


def test_batch_matches_single(client):
    single = client.post("/api/v1/score", json=PAYLOAD, headers=AUTH).json()
    batch = client.post("/api/v1/score/batch", json={"readings": [PAYLOAD, PAYLOAD]},
                        headers=AUTH).json()
    assert len(batch["results"]) == 2
    for res in batch["results"]:
        assert res["decision"] == single["decision"]
        assert res["margins"] == single["margins"]


# --- validation ------------------------------------------------------------
@pytest.mark.parametrize("patch", [
    {"product_type": "X"}, {"torque_nm": -1.0}, {"speed_rpm": 99999.0},
    {"tool_wear_min": -5.0}, {"machine_id": ""},
])
def test_invalid_payloads_are_422(client, patch):
    r = client.post("/api/v1/score", json={**PAYLOAD, **patch}, headers=AUTH)
    assert r.status_code == 422


def test_errors_never_leak_internals(client):
    r = client.post("/api/v1/score", json={**PAYLOAD, "product_type": "X"}, headers=AUTH)
    body = r.text.lower()
    for leak in ("traceback", "/app/", "site-packages", "onnxruntime", ".py\"",):
        assert leak not in body, f"response leaked {leak!r}"


# --- statelessness through the HTTP surface --------------------------------
def test_repeated_requests_are_identical(client):
    drop = {"request_id", "scored_at", "latency_ms"}
    a = client.post("/api/v1/score", json=PAYLOAD, headers=AUTH).json()
    for _ in range(20):
        client.post("/api/v1/score",
                    json={**PAYLOAD, "machine_id": "OTHER", "torque_nm": 9.0,
                          "speed_rpm": 3000.0},
                    headers=AUTH)
    b = client.post("/api/v1/score", json=PAYLOAD, headers=AUTH).json()
    assert {k: v for k, v in a.items() if k not in drop} == \
           {k: v for k, v in b.items() if k not in drop}


# --- rate limiting ---------------------------------------------------------
def test_rate_limit_returns_429(monkeypatch):
    from app.observability import TokenBucket
    from app import main as m
    with TestClient(app) as c:
        m.STATE["limiter"] = TokenBucket(per_minute=3)
        codes = [c.post("/api/v1/score", json=PAYLOAD, headers=AUTH).status_code
                 for _ in range(6)]
        assert 429 in codes
        assert codes.count(200) == 3
