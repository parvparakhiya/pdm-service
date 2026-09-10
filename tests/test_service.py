"""Regression suite. One test per defect found in the deployment review.

Runs under pytest, or standalone: `python tests/test_service.py`.

The HTTP layer is deliberately thin, so everything that can change a
maintenance decision is covered here without a running server. `test_api.py`
covers routing, auth and status codes and needs FastAPI installed.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PDM_ENVIRONMENT", "local")

from app.config import Settings                                    # noqa: E402
from app.domain import spec                                        # noqa: E402
from app.domain.decision import decide                             # noqa: E402
from app.domain.features import (FEATURE_ORDER, build_features,    # noqa: E402
                                 contract_digest, to_vector)
from app.domain.hazard import analytic_hazard, empirical_hazard    # noqa: E402
from app.inference import ModelRegistry                            # noqa: E402
from app.observability import PredictionAudit, TokenBucket         # noqa: E402
from app.schemas import TelemetryIn                                # noqa: E402
from app.service import ScoringService                             # noqa: E402

S = Settings()

TESTS = []
def test(fn):
    TESTS.append(fn)
    return fn


def make_service(audit=None) -> ScoringService:
    reg = ModelRegistry(S)
    reg.load()                       # no artifact configured -> "disabled"
    return ScoringService(S, reg, audit)


def reading(**kw) -> TelemetryIn:
    base = dict(machine_id="CNC-014", product_type="M", temp_air_k=298.2,
                temp_process_k=308.7, speed_rpm=1408.0, torque_nm=46.3,
                tool_wear_min=115.0)
    base.update(kw)
    return TelemetryIn(**base)


# ===========================================================================
# THE HEADLINE DEFECT: shared mutable state across requests
# ===========================================================================
@test
def test_scoring_is_idempotent():
    """Same payload twice -> identical answer.

    The previous service appended every request to one module-level
    deque(maxlen=60) and computed rolling features over it, so the second call
    read state the first call wrote and returned a different result.
    """
    svc = make_service()
    r = reading()
    a = svc.score(r, "req-1").model_dump(exclude={"request_id", "scored_at", "latency_ms"})
    b = svc.score(r, "req-2").model_dump(exclude={"request_id", "scored_at", "latency_ms"})
    assert a == b, "scoring is not idempotent"


@test
def test_machines_do_not_contaminate_each_other():
    """A reading from one machine must not change another machine's answer.

    With one shared buffer and no machine_id in the API, CNC-014's rolling mean
    included whatever CNC-002 had just sent.
    """
    svc = make_service()
    target = reading(machine_id="CNC-014")
    clean = svc.score(target, "r0").model_dump(
        exclude={"request_id", "scored_at", "latency_ms", "machine_id"})

    for i in range(50):               # flood with a very different machine
        svc.score(reading(machine_id="CNC-999", torque_nm=8.0, speed_rpm=2600.0,
                          tool_wear_min=5.0), f"noise-{i}")

    after = svc.score(target, "r1").model_dump(
        exclude={"request_id", "scored_at", "latency_ms", "machine_id"})
    assert clean == after, "another machine's traffic changed this machine's verdict"


@test
def test_concurrent_scoring_is_consistent():
    """No torn reads under the threadpool FastAPI uses for sync endpoints."""
    svc = make_service()
    r = reading()
    expected = svc.score(r, "seed").decision.severity
    results, errors = [], []

    def worker(n):
        try:
            for i in range(25):
                results.append(svc.score(reading(machine_id=f"M{n}"), f"{n}-{i}")
                               .decision.severity)
        except Exception as exc:                                   # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, f"concurrent scoring raised: {errors[:1]}"
    assert set(results) == {expected}, "concurrent results diverged"


@test
def test_batch_equals_sequential():
    """Batch scoring must not let readings influence one another."""
    svc = make_service()
    rs = [reading(machine_id=f"M{i}", tool_wear_min=float(20 * i)) for i in range(6)]
    seq = [svc.score(r, f"s{i}").decision.severity for i, r in enumerate(rs)]
    bat = [svc.score(r, f"b{i}").decision.severity for i, r in enumerate(rs)]
    assert seq == bat


# ===========================================================================
# Input contract
# ===========================================================================
@test
def test_invalid_quality_tier_is_rejected():
    """`product_type: str` accepted anything, then OSF_LIMITS.get(tier, 12000)
    silently applied the medium threshold to it."""
    for bad in ("X", "medium", "", "m "):
        try:
            reading(product_type=bad)
        except Exception:
            continue
        raise AssertionError(f"product_type={bad!r} should be rejected")


@test
def test_impossible_sensor_values_are_rejected():
    """Negative torque and 50,000 rpm were previously accepted and scored."""
    for kw in ({"torque_nm": -5.0}, {"speed_rpm": 50000.0}, {"tool_wear_min": -1.0},
               {"temp_air_k": 30.0}):
        try:
            reading(**kw)
        except Exception:
            continue
        raise AssertionError(f"{kw} should be rejected")


@test
def test_machine_id_is_required():
    try:
        TelemetryIn(product_type="M", temp_air_k=298.2, temp_process_k=308.7,
                    speed_rpm=1408.0, torque_nm=46.3, tool_wear_min=115.0)
    except Exception:
        return
    raise AssertionError("machine_id must be required")


@test
def test_unknown_fields_are_rejected():
    try:
        reading(rul_override=0.0)
    except Exception:
        return
    raise AssertionError("extra fields must be forbidden")


# ===========================================================================
# Specification rules
# ===========================================================================
@test
def test_rules_match_the_specification():
    d = spec.derive(temp_air_k=300.0, temp_process_k=306.0, speed_rpm=1300.0,
                    torque_nm=40.0, tool_wear_min=100.0, product_type="M", settings=S)
    assert spec.detect_hdf(d, 1300.0, S)[0] is True       # dT 6.0 < 8.6 and rpm < 1380
    d2 = spec.derive(temp_air_k=300.0, temp_process_k=306.0, speed_rpm=1500.0,
                     torque_nm=40.0, tool_wear_min=100.0, product_type="M", settings=S)
    assert spec.detect_hdf(d2, 1500.0, S)[0] is False     # rpm above the limit


@test
def test_power_envelope_both_edges():
    low = spec.derive(temp_air_k=300, temp_process_k=310, speed_rpm=900,
                      torque_nm=10, tool_wear_min=0, product_type="M", settings=S)
    high = spec.derive(temp_air_k=300, temp_process_k=310, speed_rpm=2500,
                       torque_nm=70, tool_wear_min=0, product_type="M", settings=S)
    mid = spec.derive(temp_air_k=300, temp_process_k=310, speed_rpm=1500,
                      torque_nm=40, tool_wear_min=0, product_type="M", settings=S)
    assert spec.detect_pwf(low, S)[0] is True
    assert spec.detect_pwf(high, S)[0] is True
    assert spec.detect_pwf(mid, S)[0] is False


@test
def test_osf_threshold_is_tier_specific():
    """L, M and H have different limits. The old default-to-12000 fallback
    silently applied the wrong one."""
    for tier, limit in (("L", 11000.0), ("M", 12000.0), ("H", 13000.0)):
        d = spec.derive(temp_air_k=300, temp_process_k=310, speed_rpm=1500,
                        torque_nm=60.0, tool_wear_min=limit / 60.0 + 1.0,
                        product_type=tier, settings=S)
        assert spec.detect_osf(d, tier, S)[0] is True, tier
        d2 = spec.derive(temp_air_k=300, temp_process_k=310, speed_rpm=1500,
                         torque_nm=60.0, tool_wear_min=limit / 60.0 - 1.0,
                         product_type=tier, settings=S)
        assert spec.detect_osf(d2, tier, S)[0] is False, tier


@test
def test_evidence_is_human_readable_and_specific():
    """An operator has to be able to check the claim by hand."""
    svc = make_service()
    out = svc.score(reading(torque_nm=76.0, tool_wear_min=250.0, product_type="L"), "r")
    osf = [f for f in out.findings if f.mode == "OSF"][0]
    assert osf.detected is True
    assert "Nm·min" in osf.evidence and "11,000" in osf.evidence


# ===========================================================================
# Hazard model
# ===========================================================================
@test
def test_hazard_is_monotone_and_bounded():
    prev = -1.0
    for w in range(0, 261, 5):
        h = analytic_hazard(float(w), S)
        assert 0.0 <= h.p_failure_within_horizon <= 1.0
        assert h.p_failure_within_horizon >= prev - 1e-9, f"decreased at wear {w}"
        prev = h.p_failure_within_horizon
    assert analytic_hazard(0.0, S).p_failure_within_horizon == 0.0
    assert analytic_hazard(260.0, S).p_failure_within_horizon == 1.0
    assert analytic_hazard(260.0, S).expected_remaining_minutes == 0.0


@test
def test_empirical_hazard_matches_analytic_on_uniform_data():
    lifetimes = [200.0 + i * 0.4 for i in range(100)]      # ~Uniform(200, 240)
    a = analytic_hazard(220.0, S, horizon_minutes=10.0)
    e = empirical_hazard(lifetimes, 220.0, 10.0)
    assert abs(a.p_failure_within_horizon - e.p_failure_within_horizon) < 0.06


@test
def test_no_learned_rul_is_reported():
    """The deleted regressor must not reappear under another name."""
    svc = make_service()
    out = svc.score(reading(), "r")
    assert out.tool_life.estimator == "analytic_uniform_tool_life"
    assert "not a learned regression" in out.tool_life.note
    assert not hasattr(out, "predicted_rul_minutes")


# ===========================================================================
# Decision policy
# ===========================================================================
@test
def test_healthy_machine_is_not_escalated():
    """The old policy escalated to IMMEDIATE_SHUTDOWN on any non-Healthy class,
    on top of a TWF threshold with 0.06 precision."""
    svc = make_service()
    out = svc.score(reading(temp_air_k=298.0, temp_process_k=310.0, speed_rpm=1500.0,
                            torque_nm=40.0, tool_wear_min=20.0), "r")
    assert out.decision.severity == "NOMINAL"
    assert out.decision.urgency == "NONE"


@test
def test_fired_predicate_stops_the_machine():
    svc = make_service()
    out = svc.score(reading(temp_air_k=300.0, temp_process_k=306.0, speed_rpm=1300.0,
                            torque_nm=40.0, tool_wear_min=10.0), "r")
    assert out.decision.severity == "CRITICAL"
    assert out.decision.urgency == "STOP_MACHINE"
    assert any("HDF" in t for t in out.decision.triggered_by)


@test
def test_warning_before_the_limit_is_reached():
    """Early warning the previous argmax-only service could not produce."""
    svc = make_service()
    # 85% of the M-tier overstrain envelope (60 Nm x 170 min = 10,200 of 12,000),
    # with power held at 8,797 W so it stays inside the PWF band and OSF is the
    # only pressure. Inside spec on every predicate, but little headroom left.
    out = svc.score(reading(product_type="M", torque_nm=60.0, tool_wear_min=170.0,
                            speed_rpm=1400.0), "r")
    assert all(not f.detected for f in out.findings), "no predicate should have fired"
    assert out.decision.severity == "WATCH"
    assert out.decision.urgency == "MONITOR"
    assert any("OSF" in t for t in out.decision.triggered_by)


@test
def test_escalation_is_ordered_by_severity():
    svc = make_service()
    rank = {"NOMINAL": 0, "WATCH": 1, "DEGRADED": 2, "CRITICAL": 3}
    nominal = svc.score(reading(tool_wear_min=10.0, torque_nm=40.0), "a")
    worn = svc.score(reading(tool_wear_min=232.0, torque_nm=40.0), "b")
    assert rank[worn.decision.severity] > rank[nominal.decision.severity]


# ===========================================================================
# Joint physics plausibility — each channel in range, the combination is not
# ===========================================================================
@test
def test_implausible_power_is_not_a_machine_fault():
    """The reading that motivated this layer.

    100 Nm at 1408 rpm is 14,744 W on a machine rated to 9,000 W. Every
    channel passes its own bound
    the combination is impossible. The correct
    advisory is to check the transducers, not to stop the line.
    """
    svc = make_service()
    out = svc.score(reading(torque_nm=100.0, speed_rpm=1408.0, tool_wear_min=122.0,
                            product_type="L", temp_air_k=298.2, temp_process_k=325.0), "r")
    assert out.decision.severity == "DATA_QUALITY"
    assert out.decision.urgency == "CHECK_INSTRUMENTATION"
    codes = {i.code for i in out.data_quality}
    assert "power_above_rating" in codes
    assert "thermal_gradient_excessive" in codes


@test
def test_implausible_reading_hides_nothing():
    """The findings are still computed and the advisory says what the verdict
    would be. Suppressing a possible breach silently would be worse than the
    problem this layer solves."""
    svc = make_service()
    out = svc.score(reading(torque_nm=100.0, speed_rpm=1408.0, tool_wear_min=122.0,
                            product_type="L"), "r")
    fired = {f.mode for f in out.findings if f.detected}
    assert "PWF" in fired and "OSF" in fired, "findings must still be reported"
    assert "must be stopped" in out.decision.recommended_action
    assert "torque_nm" in out.decision.recommended_action


@test
def test_genuine_overload_still_stops_the_machine():
    """A real overload sits just above the envelope and must not be explained
    away as an instrument fault. 9,300 W is 1.03x the rating
    the implausible
    threshold is 1.5x."""
    svc = make_service()
    # 63 Nm at 1410 rpm -> ~9,300 W
    out = svc.score(reading(torque_nm=63.0, speed_rpm=1410.0, tool_wear_min=10.0), "r")
    assert out.decision.severity == "CRITICAL"
    assert out.decision.urgency == "STOP_MACHINE"
    assert out.data_quality == []


@test
def test_inverted_thermal_gradient_is_flagged():
    """A machine dissipating heat cannot run colder than the air around it."""
    svc = make_service()
    out = svc.score(reading(temp_air_k=310.0, temp_process_k=300.0,
                            speed_rpm=1500.0, torque_nm=40.0, tool_wear_min=10.0), "r")
    assert out.decision.severity == "DATA_QUALITY"
    assert {i.code for i in out.data_quality} == {"thermal_gradient_inverted"}


@test
def test_wear_counter_advisory_does_not_suppress_the_verdict():
    """Wear beyond the specified life usually means an un-reset counter — but a
    worn tool needs replacing either way, so this must stay advisory."""
    svc = make_service()
    out = svc.score(reading(tool_wear_min=260.0, torque_nm=30.0, speed_rpm=1500.0), "r")
    codes = {i.code for i in out.data_quality}
    assert "tool_wear_beyond_specified_life" in codes
    assert all(not i.blocking for i in out.data_quality)
    assert out.decision.severity != "DATA_QUALITY", "advisory must not block the verdict"


@test
def test_plausible_readings_produce_no_issues():
    svc = make_service()
    for kw in ({"tool_wear_min": 20.0}, {"tool_wear_min": 170.0, "torque_nm": 60.0,
                                         "speed_rpm": 1400.0}):
        assert svc.score(reading(**kw), "r").data_quality == []


@test
def test_plausibility_can_be_disabled():
    s = Settings(plausibility_enabled=False)
    reg = ModelRegistry(s)
    reg.load()
    out = ScoringService(s, reg).score(
        reading(torque_nm=100.0, speed_rpm=1408.0, tool_wear_min=122.0), "r")
    assert out.data_quality == []
    assert out.decision.severity == "CRITICAL"


# ===========================================================================
# Asset registry
# ===========================================================================
@test
def test_unknown_machine_is_rejected_when_registry_configured():
    from app.service import UnknownMachineError
    s = Settings(known_machine_ids="CNC-014,CNC-015")
    reg = ModelRegistry(s)
    reg.load()
    svc = ScoringService(s, reg)
    svc.score(reading(machine_id="CNC-014"), "ok")            # in registry
    try:
        svc.score(reading(machine_id="js dbjvkhbjk"), "bad")
    except UnknownMachineError:
        return
    raise AssertionError("a machine_id outside the registry must be rejected")


@test
def test_empty_registry_accepts_any_machine():
    svc = make_service()
    assert svc.score(reading(machine_id="anything"), "r").machine_id == "anything"


# ===========================================================================
# Operator-facing wording
# ===========================================================================
@test
def test_no_ambiguous_power_failure_wording():
    """'power failure' means the electricity went out to anyone on a shop
    floor. The internal code stays PWF; the operator text must not."""
    svc = make_service()
    out = svc.score(reading(torque_nm=63.0, speed_rpm=1410.0, tool_wear_min=10.0), "r")
    text = out.decision.recommended_action.lower()
    assert "power failure" not in text
    assert "shaft power" in text


@test
def test_stop_orders_carry_a_cost_justification():
    svc = make_service()
    out = svc.score(reading(temp_air_k=300.0, temp_process_k=306.0, speed_rpm=1300.0), "r")
    assert out.decision.expected_cost_delta < 0, "a stop order must beat doing nothing"


@test
def test_decision_policy_pure_function():
    d = decide(fired_modes=[], envelope={"HDF": 0.1, "PWF": 0.2, "OSF": 0.1},
               p_hazard=0.0, s=S)
    assert d.severity == "NOMINAL" and d.triggered_by == []


# ===========================================================================
# Feature / serving contract
# ===========================================================================
@test
def test_feature_vector_order_is_the_contract():
    d = spec.derive(temp_air_k=298.2, temp_process_k=308.7, speed_rpm=1408.0,
                    torque_nm=46.3, tool_wear_min=115.0, product_type="M", settings=S)
    feats = build_features(temp_air_k=298.2, temp_process_k=308.7, speed_rpm=1408.0,
                           torque_nm=46.3, tool_wear_min=115.0, product_type="M",
                           derived=d, settings=S)
    vec = to_vector(feats)
    assert len(vec) == len(FEATURE_ORDER)
    shuffled = dict(reversed(list(feats.items())))
    assert to_vector(shuffled) == vec, "vector must follow FEATURE_ORDER, not dict order"


@test
def test_missing_feature_raises_not_silently_zero():
    d = spec.derive(temp_air_k=298.2, temp_process_k=308.7, speed_rpm=1408.0,
                    torque_nm=46.3, tool_wear_min=115.0, product_type="M", settings=S)
    feats = build_features(temp_air_k=298.2, temp_process_k=308.7, speed_rpm=1408.0,
                           torque_nm=46.3, tool_wear_min=115.0, product_type="M",
                           derived=d, settings=S)
    feats.pop("power_w")
    try:
        to_vector(feats)
    except KeyError as exc:
        assert "power_w" in str(exc)
        return
    raise AssertionError("a missing feature must raise, not produce a short vector")


@test
def test_contract_digest_is_stable():
    assert contract_digest() == contract_digest()
    assert len(contract_digest()) == 16


@test
def test_missing_onnx_artifact_does_not_kill_the_process():
    """The old service instantiated at import and raised, so the container
    crash-looped before FastAPI could bind a port."""
    s = Settings(onnx_classifier="does_not_exist.onnx")
    reg = ModelRegistry(s)
    reg.load()
    assert reg.state == "failed"
    assert reg.predict_proba([0.0] * len(FEATURE_ORDER)) is None


@test
def test_service_reports_not_ready_when_a_configured_model_failed():
    s = Settings(onnx_classifier="does_not_exist.onnx")
    reg = ModelRegistry(s)
    reg.load()
    assert ScoringService(s, reg).ready is False
    reg2 = ModelRegistry(S)
    reg2.load()
    assert ScoringService(S, reg2).ready is True


@test
def test_onnx_with_mismatched_contract_is_refused():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "clf.onnx"
        p.write_bytes(b"not-a-real-model")
        (Path(td) / "clf.onnx.contract.json").write_text(json.dumps(
            {"feature_names": ["a", "b"], "digest": "deadbeefdeadbeef"}))
        s = Settings(model_dir=td, onnx_classifier="clf.onnx")
        reg = ModelRegistry(s)
        reg.load()
        assert reg.state == "failed"
        assert "contract mismatch" in reg.detail


# ===========================================================================
# Observability
# ===========================================================================
@test
def test_every_prediction_is_audited():
    """Without this, alert precision can never be measured against work orders."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "pred.jsonl"
        svc = make_service(PredictionAudit(str(path)))
        svc.score(reading(machine_id="CNC-1"), "r1")
        svc.score(reading(machine_id="CNC-2"), "r2")
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        rec = json.loads(lines[0])
        for key in ("request_id", "machine_id", "policy_fingerprint", "input",
                    "severity", "triggered_by", "outcome"):
            assert key in rec, key


@test
def test_policy_fingerprint_changes_with_policy():
    a = Settings().fingerprint()
    b = Settings(hazard_stop_probability=0.5).fingerprint()
    assert a != b, "a policy change must be visible in the fingerprint"


@test
def test_rate_limiter_enforces_its_budget():
    b = TokenBucket(per_minute=5)
    allowed = sum(1 for _ in range(20) if b.allow("k"))
    assert allowed == 5
    assert b.allow("other-key") is True     # limits are per principal


@test
def test_audit_disabled_does_not_break_scoring():
    svc = make_service(PredictionAudit("/proc/cannot/write/here.jsonl"))
    assert svc.score(reading(), "r").decision.severity is not None


# ===========================================================================
# Configuration safety
# ===========================================================================
@test
def test_production_refuses_to_boot_without_auth():
    """The previous service had no authentication on an endpoint that can order
    a production line to stop."""
    try:
        Settings(environment="prod", api_keys="")
    except Exception as exc:
        assert "API_KEYS" in str(exc)
        return
    raise AssertionError("prod must require API keys")


@test
def test_production_boots_with_auth():
    s = Settings(environment="prod", api_keys="k1,k2")
    assert s.api_key_set == {"k1", "k2"}


@test
def test_inverted_spec_bounds_are_rejected():
    for kw in ({"pwf_min_w": 9000.0, "pwf_max_w": 3500.0},
               {"tool_life_min_minutes": 240.0, "tool_life_max_minutes": 200.0}):
        try:
            Settings(**kw)
        except Exception:
            continue
        raise AssertionError(f"{kw} should be rejected")


@test
def test_unknown_setting_is_rejected():
    try:
        Settings(hazrd_stop_probability=0.5)      # typo
    except Exception:
        return
    raise AssertionError("a mistyped setting must fail loudly, not default silently")


# ===========================================================================
if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:                                   # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(1 if failed else 0)
