"""Joint physics checks: is this reading possible at all?

WHY THIS LAYER EXISTS
---------------------
`app/schemas.py` validates each channel independently -- torque <= 120 Nm,
process temperature <= 325 K, and so on. Every one of those bounds can hold
while the *combination* is physically impossible.

The reading that motivated this module:

    torque 100 Nm, speed 1408 rpm  ->  shaft power 14,744 W
    air 298.2 K, process 325.0 K   ->  thermal gradient 26.8 K

Each value passes its own bound. Together they describe a machine delivering
64% more than its rated maximum power while its coolant gradient has tripled.
No spindle does that. **A drifting or failed transducer is a far more likely
explanation than a machine fault**, and the correct response is to check the
instrumentation, not to stop the line.

This distinction matters more in real plants than almost anything else in the
service. The most common cause of alarm floods is instrumentation failing, not
machines failing -- and an operator who is told to stop production because a
torque sensor drifted learns to ignore the system. That is the same
alert-fatigue failure the cost model was built to avoid, arriving through a
different door. Sensor validation is a standard layer upstream of condition
monitoring in industrial alarm design (EEMUA 191, ISA-18.2).

WHAT THIS LAYER DOES NOT DO
---------------------------
It does not hide anything. When a reading is implausible the failure-mode
findings are still computed and returned in full; only the top-level advisory
changes, and it states explicitly what the verdict would be if the
instrumentation were confirmed good. Suppressing a possible breach silently
would trade one failure mode for a worse one.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from .spec import Derived


@dataclass(frozen=True)
class PlausibilityIssue:
    code: str
    channel: str          # which instrument the operator should check
    detail: str           # the numbers, so the claim can be verified by hand
    blocking: bool        # True -> the machine state cannot be assessed


def check(derived: Derived, *, speed_rpm: float, torque_nm: float,
          tool_wear_min: float, s: Settings) -> list[PlausibilityIssue]:
    """Return every joint-physics violation found in one reading."""
    issues: list[PlausibilityIssue] = []
    if not s.plausibility_enabled:
        return issues

    # --- shaft power far above what the machine can deliver ---------------
    # PWF already fires above pwf_max. A genuine overload sits a little above
    # it; a reading well beyond the rating is an instrument, not a machine.
    ceiling = s.pwf_max_w * s.implausible_power_factor
    if derived.power_w > ceiling:
        issues.append(PlausibilityIssue(
            code="power_above_rating",
            channel="torque_nm / speed_rpm",
            detail=(f"Torque {torque_nm:.1f} Nm at {speed_rpm:.0f} rpm implies "
                    f"{derived.power_w:,.0f} W of shaft power — "
                    f"{derived.power_w / s.pwf_max_w:.0%} of the "
                    f"{s.pwf_max_w:,.0f} W rating. The machine cannot deliver this."),
            blocking=True))

    # --- thermal gradient inverted ----------------------------------------
    # The process dissipates heat into the air, so process temperature is
    # always above air temperature. Below it means a swapped or failed probe.
    if derived.temp_delta_k < s.implausible_delta_t_min_k:
        issues.append(PlausibilityIssue(
            code="thermal_gradient_inverted",
            channel="temp_process_k / temp_air_k",
            detail=(f"Process temperature is {abs(derived.temp_delta_k):.1f} K "
                    f"below air temperature. A machine dissipating heat cannot run "
                    f"colder than its surroundings — suspect swapped or failed probes."),
            blocking=True))

    # --- thermal gradient implausibly large -------------------------------
    elif derived.temp_delta_k > s.implausible_delta_t_max_k:
        issues.append(PlausibilityIssue(
            code="thermal_gradient_excessive",
            channel="temp_process_k / temp_air_k",
            detail=(f"Thermal gradient {derived.temp_delta_k:.1f} K exceeds the "
                    f"{s.implausible_delta_t_max_k:.0f} K plausible maximum for this "
                    f"machine — suspect a drifting temperature channel."),
            blocking=True))

    # --- tool wear beyond its specified life ------------------------------
    # ADVISORY, not blocking. The usual cause is a wear counter that was not
    # reset after a tool change. But a worn tool still needs replacing either
    # way, so this must not suppress the tool-wear verdict.
    if tool_wear_min > s.tool_life_max_minutes:
        issues.append(PlausibilityIssue(
            code="tool_wear_beyond_specified_life",
            channel="tool_wear_min",
            detail=(f"Tool wear {tool_wear_min:.0f} min exceeds the specified maximum "
                    f"life of {s.tool_life_max_minutes:.0f} min. Most often a wear "
                    f"counter that was not reset at the last tool change — verify "
                    f"before trusting the remaining-life figure."),
            blocking=False))

    return issues


def blocking(issues: list[PlausibilityIssue]) -> list[PlausibilityIssue]:
    return [i for i in issues if i.blocking]
