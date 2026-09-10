"""The scoring service. All logic lives here; the HTTP layer is a thin adapter.

Splitting it this way is deliberate. Every rule that can change a maintenance
decision is reachable from a plain function call, so it can be unit-tested,
replayed against the audit log, and run in a batch job without standing up a web
server. `app/main.py` only does routing, auth and serialisation.

The service is STATELESS. Scoring the same reading twice returns byte-identical
results; two replicas behind a load balancer always agree; a restart changes
nothing. That is a direct consequence of dropping the shared rolling buffer --
see app/domain/features.py for why that buffer was unsalvageable.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from .config import Settings
from .domain import decision as dpolicy
from .domain import plausibility, spec
from .domain.features import build_features, to_vector
from .domain.hazard import analytic_hazard
from .inference import ModelRegistry
from .observability import METRICS, PredictionAudit
from .schemas import (DataQualityIssue, Decision, FailureMode, FindingSource,
                      ModeFinding, PredictionOut, TelemetryIn, ToolLife)

log = logging.getLogger("pdm.service")


class UnknownMachineError(ValueError):
    """Raised when machine_id is absent from a configured asset registry."""


class ScoringService:
    def __init__(self, settings: Settings, registry: ModelRegistry,
                 audit: PredictionAudit | None = None):
        self.settings = settings
        self.registry = registry
        self.audit = audit
        self._fingerprint = settings.fingerprint()

    # ------------------------------------------------------------------
    @property
    def fingerprint(self) -> str:
        """Hash of the policy that produced a prediction. Stamped on every
        response and audit record so any decision can be traced to its config."""
        return self._fingerprint

    @property
    def ready(self) -> bool:
        """The rule engine and hazard model are closed form and always
        available, so the service is ready unless a *configured* ONNX model
        failed to load. A model that was never configured is not a fault."""
        return self.registry.state != "failed"

    # ------------------------------------------------------------------
    def score(self, reading: TelemetryIn, request_id: str) -> PredictionOut:
        t0 = time.perf_counter()
        s = self.settings

        '''A typo in machine_id creates a phantom asset in the audit log, and
        those alerts then vanish from the work-order reconciliation. Only
        enforced when a registry is configured.'''
        registry = s.machine_registry
        if registry and reading.machine_id not in registry:
            raise UnknownMachineError(reading.machine_id)

        derived = spec.derive(
            temp_air_k=reading.temp_air_k, temp_process_k=reading.temp_process_k,
            speed_rpm=reading.speed_rpm, torque_nm=reading.torque_nm,
            tool_wear_min=reading.tool_wear_min, product_type=reading.product_type,
            settings=s)

        # ---- deterministic modes: exact, not probabilistic ---------------
        findings: list[ModeFinding] = []
        fired: list[str] = []

        env = spec.envelope_used(derived, reading.speed_rpm, s)

        hdf_fired, hdf_why = spec.detect_hdf(derived, reading.speed_rpm, s)
        pwf_fired, pwf_why = spec.detect_pwf(derived, s)
        osf_fired, osf_why = spec.detect_osf(derived, reading.product_type, s)

        rule_modes: tuple[tuple[FailureMode, bool, str], ...] = (
            ("HDF", hdf_fired, hdf_why),
            ("PWF", pwf_fired, pwf_why),
            ("OSF", osf_fired, osf_why))
        for mode, fired_flag, why in rule_modes:
            findings.append(ModeFinding(
                mode=mode, detected=fired_flag, source="specification_rule",
                confidence=1.0,              # the predicate is the definition
                envelope_used=round(env[mode], 4), evidence=why))
            if fired_flag:
                fired.append(mode)

        # ---- tool wear: hazard over an explicit horizon -------------------
        hz = analytic_hazard(reading.tool_wear_min, s)
        twf_source: FindingSource = "analytic_hazard"
        twf_conf = hz.p_failure_within_horizon
        twf_evidence = (
            f"Tool wear {reading.tool_wear_min:.0f} min against a specified life of "
            f"{s.tool_life_min_minutes:.0f}–{s.tool_life_max_minutes:.0f} min: "
            f"{twf_conf:.0%} chance of failure within the next "
            f"{hz.horizon_minutes:.0f} minutes, "
            f"{hz.expected_remaining_minutes:.0f} min expected remaining.")

        # An ONNX model is consulted only if one earned deployment. When
        # present it *replaces* the hazard probability for its own mode and
        # says so, rather than being silently blended in.
        if self.registry.state == "loaded" and self.registry.classifier is not None:
            if self.registry.classifier.mode == "TWF":
                try:
                    feats = build_features(
                        temp_air_k=reading.temp_air_k, temp_process_k=reading.temp_process_k,
                        speed_rpm=reading.speed_rpm, torque_nm=reading.torque_nm,
                        tool_wear_min=reading.tool_wear_min,
                        product_type=reading.product_type, derived=derived, settings=s)
                    p = self.registry.predict_proba(to_vector(feats))
                    if p is not None:
                        twf_conf, twf_source = float(p), "onnx_model"
                        twf_evidence = (
                            f"Calibrated model probability {twf_conf:.1%} of tool-wear "
                            f"failure. Baseline hazard for this wear level was "
                            f"{hz.p_failure_within_horizon:.1%}.")
                except Exception as exc:                      # noqa: BLE001
                    # Never fail a maintenance decision because an optional
                    # model misbehaved: fall back to the closed-form hazard,
                    # which is always available, and record it.
                    log.error("onnx scoring failed, falling back to analytic hazard",
                              exc_info=exc, extra={"machine_id": reading.machine_id})

        findings.append(ModeFinding(
            mode="TWF", detected=twf_conf >= s.hazard_stop_probability,
            source=twf_source, confidence=round(min(max(twf_conf, 0.0), 1.0), 4),
            envelope_used=round(reading.tool_wear_min / s.tool_life_max_minutes, 4),
            evidence=twf_evidence))

        # ---- joint physics: is this reading possible at all? --------------
        issues = plausibility.check(
            derived, speed_rpm=reading.speed_rpm, torque_nm=reading.torque_nm,
            tool_wear_min=reading.tool_wear_min, s=s)
        if plausibility.blocking(issues):
            log.warning("implausible reading, machine state not assessed",
                        extra={"machine_id": reading.machine_id,
                               "codes": [i.code for i in issues if i.blocking]})

        # ---- policy ------------------------------------------------------
        d = dpolicy.decide(fired_modes=fired, envelope=env, p_hazard=twf_conf, s=s,
                           plausibility_issues=issues)
        marg = {k: round(v, 4) for k, v in spec.margins(derived, reading.speed_rpm, s).items()}
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        out = PredictionOut(
            request_id=request_id,
            machine_id=reading.machine_id,
            policy_fingerprint=self._fingerprint,
            findings=findings,
            tool_life=ToolLife(
                estimator="analytic_uniform_tool_life",
                horizon_minutes=hz.horizon_minutes,
                p_failure_within_horizon=round(hz.p_failure_within_horizon, 4),
                expected_remaining_minutes=round(hz.expected_remaining_minutes, 2)),
            decision=Decision(
                severity=d.severity, urgency=d.urgency, recommended_action=d.action,
                triggered_by=d.triggered_by,
                expected_cost_delta=round(d.expected_cost_delta, 2)),
            data_quality=[DataQualityIssue(code=i.code, channel=i.channel,
                                           detail=i.detail, blocking=i.blocking)
                          for i in issues],
            margins=marg,
            latency_ms=round(elapsed_ms, 3),
        )

        METRICS.observe_decision(d.severity, d.urgency, fired, twf_conf)
        self._audit(reading, out)
        return out

    # ------------------------------------------------------------------
    def _audit(self, reading: TelemetryIn, out: PredictionOut) -> None:
        if self.audit is None:
            return
        self.audit.record({
            "ts": datetime.now(timezone.utc).isoformat(),
            "request_id": out.request_id,
            "machine_id": reading.machine_id,
            "policy_fingerprint": out.policy_fingerprint,
            "input": reading.model_dump(mode="json"),
            "severity": out.decision.severity,
            "urgency": out.decision.urgency,
            "triggered_by": out.decision.triggered_by,
            "findings": [{"mode": f.mode, "detected": f.detected,
                          "source": f.source, "confidence": f.confidence}
                         for f in out.findings],
            "margins": out.margins,
            "data_quality": [i.code for i in out.data_quality],
            "latency_ms": out.latency_ms,
            # Filled in later by the work-order reconciliation job. This is the
            # column that makes precision measurable against reality.
            "outcome": None,
        })
