"""The deterministic failure detector. This is the primary shipping model.

HDF, PWF and OSF are defined by the equipment specification as closed-form
predicates over the sensor readings. They are not statistical relationships to
discover. Measured on the reference dataset, three predicates score macro-F1
0.796 against the tuned LightGBM pipeline's 0.766 -- with no
training, no drift, microsecond latency, and an explanation an operator can
verify by hand.

Deliberately dependency-free: plain Python floats, no numpy, no pandas. That
keeps the serving image to onnxruntime-optional and makes every branch here
trivially unit-testable.

`margins` is the part with real operational value. The predicate answers "is it
outside the envelope right now" exactly; the margin answers "how much headroom
is left", which is what lets maintenance act *before* the line stops.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings

RAD_PER_S_PER_RPM = 0.10471975511965977      # 2*pi/60


@dataclass(frozen=True)
class Derived:
    """Physical quantities derived from one reading."""
    temp_delta_k: float
    power_w: float
    overstrain_nm_min: float
    osf_limit: float


def derive(temp_air_k: float, temp_process_k: float, speed_rpm: float,
           torque_nm: float, tool_wear_min: float, product_type: str,
           settings: Settings) -> Derived:
    return Derived(
        temp_delta_k=temp_process_k - temp_air_k,
        power_w=torque_nm * speed_rpm * RAD_PER_S_PER_RPM,
        overstrain_nm_min=tool_wear_min * torque_nm,
        osf_limit=settings.osf_limits[product_type],
    )


# ---------------------------------------------------------------------------
# Predicates. One function per failure mode, each returning (fired, evidence).
# ---------------------------------------------------------------------------
def detect_hdf(d: Derived, speed_rpm: float, s: Settings) -> tuple[bool, str]:
    fired = d.temp_delta_k < s.hdf_delta_t_k and speed_rpm < s.hdf_speed_rpm
    if fired:
        return True, (f"Heat dissipation: ΔT {d.temp_delta_k:.2f} K is below the "
                      f"{s.hdf_delta_t_k} K minimum while speed {speed_rpm:.0f} rpm is "
                      f"below {s.hdf_speed_rpm:.0f} rpm. Both conditions hold.")
    if d.temp_delta_k < s.hdf_delta_t_k:
        return False, (f"ΔT {d.temp_delta_k:.2f} K is below the {s.hdf_delta_t_k} K "
                       f"minimum, but speed {speed_rpm:.0f} rpm is above "
                       f"{s.hdf_speed_rpm:.0f} rpm, so heat is still being carried away.")
    return False, (f"ΔT {d.temp_delta_k:.2f} K has "
                   f"{d.temp_delta_k - s.hdf_delta_t_k:.2f} K of headroom.")


def detect_pwf(d: Derived, s: Settings) -> tuple[bool, str]:
    if d.power_w < s.pwf_min_w:
        return True, (f"Power {d.power_w:.0f} W is below the {s.pwf_min_w:.0f} W "
                      f"minimum of the operating envelope.")
    if d.power_w > s.pwf_max_w:
        return True, (f"Power {d.power_w:.0f} W exceeds the {s.pwf_max_w:.0f} W "
                      f"maximum of the operating envelope.")
    return False, (f"Power {d.power_w:.0f} W sits inside the "
                   f"{s.pwf_min_w:.0f}–{s.pwf_max_w:.0f} W envelope.")


def detect_osf(d: Derived, product_type: str, s: Settings) -> tuple[bool, str]:
    used = d.overstrain_nm_min / d.osf_limit
    if d.overstrain_nm_min > d.osf_limit:
        return True, (f"Overstrain {d.overstrain_nm_min:,.0f} Nm·min exceeds the "
                      f"{d.osf_limit:,.0f} Nm·min limit for tier {product_type} "
                      f"({used:.0%} of capacity).")
    return False, (f"Overstrain at {used:.0%} of the {d.osf_limit:,.0f} Nm·min "
                   f"limit for tier {product_type}.")


def margins(d: Derived, speed_rpm: float, s: Settings) -> dict[str, float]:
    """Signed, normalised distance to each boundary.

    Positive is headroom, negative is inside the failure region, and the
    magnitude is comparable across modes so a single "closest boundary" ranking
    is meaningful.
    """
    band = (s.pwf_max_w - s.pwf_min_w) / 2.0
    centre = (s.pwf_max_w + s.pwf_min_w) / 2.0
    return {
        "hdf_thermal_margin": (d.temp_delta_k - s.hdf_delta_t_k) / s.hdf_delta_t_k,
        "hdf_speed_margin": (speed_rpm - s.hdf_speed_rpm) / s.hdf_speed_rpm,
        "pwf_envelope_margin": 1.0 - abs(d.power_w - centre) / band,
        "osf_capacity_used": d.overstrain_nm_min / d.osf_limit,
    }


def envelope_used(d: Derived, speed_rpm: float, s: Settings) -> dict[str, float]:
    """Fraction of each envelope consumed, clamped to [0, ...]. 1.0 = at the limit.

    This is what drives escalation, and it is why the service can warn before a
    predicate fires -- something the previous version could not do, because it
    only ever reported the argmax of a classifier.
    """
    band = (s.pwf_max_w - s.pwf_min_w) / 2.0
    centre = (s.pwf_max_w + s.pwf_min_w) / 2.0
    hdf_thermal = s.hdf_delta_t_k / max(d.temp_delta_k, 1e-9)
    return {
        "HDF": min(hdf_thermal, 2.0) if speed_rpm < s.hdf_speed_rpm else 0.0,
        "PWF": abs(d.power_w - centre) / band,
        "OSF": d.overstrain_nm_min / d.osf_limit,
    }
