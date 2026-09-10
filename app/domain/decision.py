"""Turning findings into an action, under an explicit cost model.

WHAT WAS WRONG BEFORE
---------------------
The previous policy was:

    if rul <= 15 or osf_ratio >= 0.95 or failure_status != "Healthy":
        -> CRITICAL / IMMEDIATE_SHUTDOWN

Three problems, compounding.

*   `failure_status != "Healthy"` escalated on *any* non-healthy class. Combined
    with `if probs[1] > 0.12: pred_class_idx = 1` -- a threshold whose measured
    precision was 0.06 -- the service ordered a line stop on roughly one in
    twenty healthy machines. Alert fatigue is not a UX problem here; a stop
    order that is wrong 94% of the time gets disabled, and then the 6% that were
    real go unactioned too.
*   `rul <= 15` used a regressor with out-of-sample R^2 = -0.104.
*   Nothing was expressed in money, so nobody could argue with it.

WHAT REPLACES IT
----------------
Escalation comes from two sources that are both defensible:

*   a specification predicate that has actually fired (exact, not probabilistic),
    or an envelope that is measurably close to its limit; and
*   a tool-wear hazard with an explicit horizon: "P(failure in the next 10
    minutes)", not a point estimate of remaining life.

Every alert reports its expected cost delta against doing nothing, so the
thresholds can be argued about with numbers in a review rather than adjusted by
feel in a hotfix.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings


@dataclass(frozen=True)
class DecisionResult:
    severity: str
    urgency: str
    action: str
    triggered_by: list[str]
    expected_cost_delta: float


# Operator-facing wording. The dataset calls PWF "power failure", but to
# someone on the shop floor that means the electricity went out -- a completely
# different response. The internal code stays PWF; the words an operator reads
# say what actually happened.
MODE_NAMES = {
    "TWF": "tool wear",
    "HDF": "heat dissipation — coolant gradient too low at low speed",
    "PWF": "shaft power outside its envelope",
    "OSF": "mechanical overstrain",
}


def _expected_cost_delta(p_failure: float, acting: bool, s: Settings) -> float:
    """Cost of acting minus cost of not acting, for one decision.

    not acting : p * cost_missed_failure
    acting     : cost_planned_stop + (1 - p) * cost_false_alarm
    """
    do_nothing = p_failure * s.cost_missed_failure
    if not acting:
        return 0.0
    act = s.cost_planned_stop + (1.0 - p_failure) * s.cost_false_alarm
    return act - do_nothing


def decide(*, fired_modes: list[str], envelope: dict[str, float],
           p_hazard: float, s: Settings,
           plausibility_issues: list | None = None) -> DecisionResult:
    triggered: list[str] = []

    # ---- 0. is the reading physically possible at all? -------------------
    # This precedes everything, because a verdict computed from an impossible
    # reading is not a verdict. The findings are still returned in full and
    # the advisory says what the answer would be if the instruments check out,
    # so nothing is suppressed -- but the machine is not stopped on the word
    # of a transducer that is contradicting itself.
    blocking = [i for i in (plausibility_issues or []) if getattr(i, "blocking", False)]
    if blocking:
        channels = sorted({i.channel for i in blocking})
        would_be = (" If the instrumentation checks out, this reading is a breach of "
                    + ", ".join(MODE_NAMES.get(m, m) for m in fired_modes)
                    + " and the machine must be stopped." if fired_modes else
                    " No specification limit was breached by the reading as given.")
        return DecisionResult(
            severity="DATA_QUALITY", urgency="CHECK_INSTRUMENTATION",
            action=("Machine state could not be assessed: the reading is not physically "
                    "self-consistent. Verify " + " and ".join(channels) + "." + would_be),
            triggered_by=[f"{i.code}: {i.detail}" for i in blocking],
            # Undefined, not zero-risk: the state is unknown, so there is no
            # expected cost to compare against. Reporting a number here would
            # imply a confidence the service does not have.
            expected_cost_delta=0.0)

    # ---- 1. a specification predicate has actually fired -----------------
    # These are exact, not probabilistic: the machine is outside its envelope
    # right now. Certainty is 1.0, so acting is essentially always correct.
    if fired_modes:
        for m in fired_modes:
            triggered.append(f"{m}: specification limit exceeded")
        return DecisionResult(
            severity="CRITICAL", urgency="STOP_MACHINE",
            action=("Stop the machine. Specification limit exceeded — "
                    + "; ".join(MODE_NAMES.get(m, m) for m in fired_modes)
                    + ". This is a measured breach of the operating envelope, not a forecast."),
            triggered_by=triggered,
            expected_cost_delta=_expected_cost_delta(1.0, acting=True, s=s))

    # ---- 2. tool-wear hazard over an explicit horizon --------------------
    if p_hazard >= s.hazard_stop_probability:
        triggered.append(
            f"TWF: P(failure within {s.hazard_horizon_minutes:.0f} min) = {p_hazard:.0%}")
        return DecisionResult(
            severity="CRITICAL", urgency="STOP_MACHINE",
            action=(f"Replace the tool before the next cycle: {p_hazard:.0%} chance of "
                    f"tool-wear failure within {s.hazard_horizon_minutes:.0f} minutes."),
            triggered_by=triggered,
            expected_cost_delta=_expected_cost_delta(p_hazard, acting=True, s=s))

    # ---- 3. envelope pressure: warn before the predicate fires -----------
    critical = {m: u for m, u in envelope.items() if u >= s.margin_critical_fraction}
    warning = {m: u for m, u in envelope.items()
               if s.margin_warning_fraction <= u < s.margin_critical_fraction}

    if critical:
        for m, u in sorted(critical.items(), key=lambda kv: -kv[1]):
            triggered.append(f"{m}: {u:.0%} of operating envelope consumed")
        worst = max(critical.values())
        return DecisionResult(
            severity="DEGRADED", urgency="SCHEDULE_MAINTENANCE",
            action=(f"Schedule maintenance this shift. {worst:.0%} of the "
                    f"{max(critical, key=critical.get)} envelope is consumed; the machine is "
                    "still inside spec but has little headroom left."),
            triggered_by=triggered,
            expected_cost_delta=_expected_cost_delta(max(p_hazard, 0.25), acting=True, s=s))

    if p_hazard >= s.hazard_schedule_probability:
        triggered.append(
            f"TWF: P(failure within {s.hazard_horizon_minutes:.0f} min) = {p_hazard:.0%}")
        return DecisionResult(
            severity="DEGRADED", urgency="SCHEDULE_MAINTENANCE",
            action=("Queue a tool change at the next planned window: "
                    f"{p_hazard:.0%} chance of tool-wear failure within "
                    f"{s.hazard_horizon_minutes:.0f} minutes."),
            triggered_by=triggered,
            expected_cost_delta=_expected_cost_delta(p_hazard, acting=True, s=s))

    if warning:
        for m, u in sorted(warning.items(), key=lambda kv: -kv[1]):
            triggered.append(f"{m}: {u:.0%} of operating envelope consumed")
        return DecisionResult(
            severity="WATCH", urgency="MONITOR",
            action=("No action required. Trending toward the "
                    f"{max(warning, key=warning.get)} limit; keep it on the watch list."),
            triggered_by=triggered,
            expected_cost_delta=0.0)

    # ---- 4. nominal ------------------------------------------------------
    return DecisionResult(
        severity="NOMINAL", urgency="NONE",
        action="Operating within specification on every monitored failure mode.",
        triggered_by=[], expected_cost_delta=0.0)
