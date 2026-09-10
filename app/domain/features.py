"""Feature vector for the optional ONNX model. Stateless by construction.

WHY THERE IS NO BUFFER HERE
---------------------------
The previous service kept a module-level `StatefulFeatureBuffer` -- a single
`deque(maxlen=60)` shared by the whole process -- and appended every incoming
request to it, then computed `rolling(10/30/60)` statistics over that deque.
Four separate failures came out of that one object:

1.  ONE BUFFER, ALL MACHINES. There was no machine identity in the API at all,
    so a reading from CNC-014 was averaged together with readings from every
    other machine that happened to call in the last 60 requests.

2.  NOT IDEMPOTENT. The same payload posted twice returns different answers,
    because the first call mutated the state the second one reads. That breaks
    retries, replay, and every form of debugging.

3.  NOT HORIZONTALLY SCALABLE OR THREAD-SAFE. Two replicas behind a load
    balancer hold different buffers and disagree. Within one replica, FastAPI
    runs `def` endpoints in a threadpool, so concurrent requests interleave
    reads and writes of the same deque with no lock.

4.  TOTAL TRAIN/SERVE SKEW. On the first request the buffer holds one row, so
    every rolling mean equals the current value and every rolling std is
    exactly 0 -- values the model never saw in training, where the same columns
    were computed over 10,000 rows.

The underlying issue is that those rolling features were never legitimate: the
training data has no machine identity and no clock, so a 60-row window averaged
60 unrelated products, and ablation showed removing all twelve of them changed
macro-F1 by -0.002 -- no measurable contribution. The fix is not a better buffer. It is to drop the features,
which is what the v2 pipeline did -- leaving a feature set that is computable
from a single reading, which is what makes this module stateless.

If genuine per-machine history becomes available, it belongs in a feature store
keyed by machine_id with point-in-time correctness, not in a process-local deque.
"""
from __future__ import annotations

from ..config import Settings
from .spec import Derived, margins

# The order here IS the serving contract. It is verified against the digest
# recorded at training time before any ONNX session is used -- so a reordered
# column fails loudly at startup instead of producing silently wrong
# predictions, which is what a positional `FloatTensorType([None, 21])` plus a
# feature list in a pickle allowed.
FEATURE_ORDER: tuple[str, ...] = (
    "temp_air_k", "temp_process_k", "speed_rpm", "torque_nm", "tool_wear_min",
    "temp_delta_k", "power_w", "overstrain_nm_min", "quality_code",
    "torque_per_rpm", "power_per_kelvin", "tool_life_fraction",
    "hdf_thermal_margin", "hdf_speed_margin", "pwf_envelope_margin",
    "osf_capacity_used",
)

QUALITY_CODE = {"L": 0.0, "M": 1.0, "H": 2.0}


def build_features(*, temp_air_k: float, temp_process_k: float, speed_rpm: float,
                   torque_nm: float, tool_wear_min: float, product_type: str,
                   derived: Derived, settings: Settings) -> dict[str, float]:
    """Pure function of one reading. Same input, same output, always."""
    life_mid = 0.5 * (settings.tool_life_min_minutes + settings.tool_life_max_minutes)
    feats = {
        "temp_air_k": temp_air_k,
        "temp_process_k": temp_process_k,
        "speed_rpm": speed_rpm,
        "torque_nm": torque_nm,
        "tool_wear_min": tool_wear_min,
        "temp_delta_k": derived.temp_delta_k,
        "power_w": derived.power_w,
        "overstrain_nm_min": derived.overstrain_nm_min,
        "quality_code": QUALITY_CODE[product_type],
        "torque_per_rpm": torque_nm / max(speed_rpm, 1e-9),
        "power_per_kelvin": derived.power_w / max(derived.temp_delta_k, 0.5),
        "tool_life_fraction": tool_wear_min / life_mid,
    }
    feats.update(margins(derived, speed_rpm, settings))
    return feats


def to_vector(feats: dict[str, float]) -> list[float]:
    """Order by the contract, never by dict insertion order."""
    missing = [c for c in FEATURE_ORDER if c not in feats]
    if missing:
        raise KeyError(f"feature contract violated, missing: {missing}")
    return [float(feats[c]) for c in FEATURE_ORDER]


def contract_digest() -> str:
    import hashlib
    return hashlib.sha256("|".join(FEATURE_ORDER).encode()).hexdigest()[:16]
