"""Tool-wear failure as a hazard. There is no learned RUL model, on purpose.

The previous service loaded `rul_model.onnx` and surfaced its output as
`predicted_rul_minutes`, then used it to order line stops at <= 15 minutes. That
model's training target was

    RUL = clip(limit(quality) - tool_wear, 0, 120)

which is one subtraction, and its reported R^2 = 0.845 came from neighbouring
rows: under contiguous-block cross-validation the same model scores -0.104,
worse than predicting the mean. Shipping it means shutting down production lines
on a number that does not generalise.

What is true instead: tool life is drawn on [200, 240] minutes independently of
every sensor reading. Given survival to wear w, remaining life is exactly

    L | L > w  ~  Uniform(max(w, 200), 240)

so the hazard is closed form. No training data, no artifact, no drift, and no
model can beat it -- the randomness is in the process, not in our ignorance.

`empirical_hazard` is the upgrade path: once the plant has real run-to-failure
records, swap the uniform for the observed survival function. That, not more
boosting, is what improves this number.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings


@dataclass(frozen=True)
class HazardEstimate:
    p_failure_within_horizon: float
    expected_remaining_minutes: float
    horizon_minutes: float
    estimator: str


def analytic_hazard(tool_wear_min: float, s: Settings,
                    horizon_minutes: float | None = None) -> HazardEstimate:
    horizon = s.hazard_horizon_minutes if horizon_minutes is None else horizon_minutes
    lo, hi = s.tool_life_min_minutes, s.tool_life_max_minutes
    w = float(tool_wear_min)

    if w >= hi:                       # past the maximum specified life
        return HazardEstimate(1.0, 0.0, horizon, "analytic_uniform_tool_life")

    a = max(w, lo)                    # lower edge of the surviving life support
    width = max(hi - a, 1e-9)
    p = (w + horizon - a) / width
    p = min(max(p, 0.0), 1.0)
    expected = max(0.5 * (a + hi) - w, 0.0)
    return HazardEstimate(p, expected, horizon, "analytic_uniform_tool_life")


def empirical_hazard(observed_lifetimes: list[float], tool_wear_min: float,
                     horizon_minutes: float) -> HazardEstimate:
    """Survival-function hazard for real plant data.

    Requires run-to-failure records -- exactly the data AI4I lacks. Wire this in
    once the historian carries machine_id, timestamp and replacement events.
    """
    lt = sorted(float(x) for x in observed_lifetimes)
    if not lt:
        raise ValueError("no observed lifetimes supplied")
    w = float(tool_wear_min)

    def survival(x: float) -> float:
        beyond = sum(1 for v in lt if v > x)
        return beyond / len(lt)

    s_now = survival(w)
    s_then = survival(w + horizon_minutes)
    p = 1.0 - (s_then / s_now) if s_now > 0 else 1.0
    remaining = [v - w for v in lt if v > w]
    expected = sum(remaining) / len(remaining) if remaining else 0.0
    return HazardEstimate(min(max(p, 0.0), 1.0), max(expected, 0.0),
                          horizon_minutes, "empirical_survival")
