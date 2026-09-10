"""API request/response contracts and edge schema enforcement.

Defines strict, strongly-typed Pydantic models for all inference payloads. 
This module guarantees interface integrity through rigorous input validation 
and explicit, versioned output structuring:

*   TYPE SAFETY & BOUNDARY ENFORCEMENT: Restricts product tiers to explicit literals 
    (`Literal["L", "M", "H"]`) and enforces physical range bounds on incoming sensor telemetry 
    at the edge, preventing silent fallbacks and unphysical scoring.
*   STRONGLY-TYPED RESPONSES: Replaces ambiguous, untyped dictionaries with explicit 
    schema models for class probabilities and root-cause telemetry, providing a stable 
    contract for downstream consumers.
*   STATE INTEGRITY & ISOLATION: Requires explicit `machine_id` propagation on all payloads, 
    ensuring proper request tracing and preventing multi-machine state cross-contamination.
*   TRANSPARENT ESTIMATION PROVENANCE: Disambiguates predictive outputs, explicitly labelling 
    remaining life calculations with their underlying estimator provenance rather than 
    presenting them as black-box learned quantities.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import CONTRACT_VERSION, QualityTier

Severity = Literal["NOMINAL", "WATCH", "DEGRADED", "CRITICAL", "DATA_QUALITY"]
Urgency = Literal["NONE", "MONITOR", "SCHEDULE_MAINTENANCE", "STOP_MACHINE",
                  "CHECK_INSTRUMENTATION"]
FailureMode = Literal["TWF", "HDF", "PWF", "OSF"]


class TelemetryIn(BaseModel):
    """One reading from one machine. Stateless: this is everything the service
    needs. There is deliberately no history parameter -- see app/domain/features.py."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "machine_id": "CNC-014", "product_type": "M", "temp_air_k": 298.2,
        "temp_process_k": 308.7, "speed_rpm": 1408.0, "torque_nm": 46.3,
        "tool_wear_min": 115.0}]})

    machine_id: str = Field(..., min_length=1, max_length=64,
                            description="Stable identifier for the physical machine.")
    product_type: QualityTier = Field(..., description="Tool quality tier: L, M or H.")
    temp_air_k: float = Field(..., ge=290.0, le=315.0, description="Air temperature (K).")
    temp_process_k: float = Field(..., ge=295.0, le=325.0, description="Process temperature (K).")
    speed_rpm: float = Field(..., ge=500.0, le=4000.0, description="Rotational speed (rpm).")
    torque_nm: float = Field(..., ge=0.0, le=120.0, description="Torque (Nm).")
    tool_wear_min: float = Field(..., ge=0.0, le=300.0, description="Tool wear (minutes).")
    observed_at: datetime | None = Field(
        default=None, description="Reading timestamp; defaults to receipt time.")


class BatchTelemetryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    readings: list[TelemetryIn] = Field(..., min_length=1, max_length=500)


class ModeFinding(BaseModel):
    """One failure mode, with the evidence that produced the verdict."""
    mode: FailureMode
    detected: bool
    source: Literal["specification_rule", "analytic_hazard", "onnx_model"]
    # Exact for rules; a probability for the hazard and any model.
    confidence: float = Field(..., ge=0.0, le=1.0)
    envelope_used: float | None = Field(
        None, description="Fraction of the operating envelope consumed (1.0 = at the limit).")
    evidence: str = Field(..., description="Human-readable statement of why.")


class ToolLife(BaseModel):
    estimator: Literal["analytic_uniform_tool_life", "empirical_survival"] = (
        "analytic_uniform_tool_life")
    horizon_minutes: float
    p_failure_within_horizon: float = Field(..., ge=0.0, le=1.0)
    expected_remaining_minutes: float = Field(..., ge=0.0)
    note: str = (
        "Closed form from the specified tool-life window; not a learned "
        "regression. No learned RUL model is deployed -- see docs/DECISIONS.md.")


class DataQualityIssue(BaseModel):
    """A joint-physics violation: each channel is in range, the combination is not."""
    code: str
    channel: str = Field(..., description="Which instrument to check.")
    detail: str = Field(..., description="The numbers, so the claim can be verified by hand.")
    blocking: bool = Field(..., description="True when the machine state could not be assessed.")


class Decision(BaseModel):
    severity: Severity
    urgency: Urgency
    recommended_action: str
    triggered_by: list[str] = Field(default_factory=list)
    expected_cost_delta: float = Field(
        ..., description="Expected cost of acting on this alert minus doing nothing, "
                         "under the configured cost model. Negative is good.")


class PredictionOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = CONTRACT_VERSION
    request_id: str
    machine_id: str
    scored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    policy_fingerprint: str

    findings: list[ModeFinding]
    tool_life: ToolLife
    decision: Decision
    # Populated when a reading is not physically self-consistent. Findings are
    # still returned in full -- nothing is hidden; only the advisory changes.
    data_quality: list[DataQualityIssue] = Field(default_factory=list)
    margins: dict[str, float] = Field(
        ..., description="Signed distance to each specification boundary. "
                         ">0 is headroom, <0 is inside the failure region.")
    latency_ms: float


class BatchPredictionOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[PredictionOut]


class HealthOut(BaseModel):
    status: Literal["healthy"] = "healthy"
    service: str
    version: str
    environment: str


class ReadinessOut(BaseModel):
    ready: bool
    detail: str
    rule_engine: bool
    hazard_model: bool
    onnx_classifier: Literal["loaded", "disabled", "failed"]
    contract_digest: str | None = None
    policy_fingerprint: str


class ErrorOut(BaseModel):
    error: str
    code: Literal["validation_error", "unauthorized", "rate_limited",
                  "not_ready", "internal_error", "unknown_machine"]
    request_id: str
