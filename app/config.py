"""Environment-driven configuration and runtime settings validation.

Enforces strict startup validation to guarantee production reliability and security:

* Fail-Fast Validation: Environment variables and configuration parameters 
  are validated at startup, preventing silent fallbacks and misconfigurations 
  that could compromise operational decisions.
* Explicit Governance: Specification constants and business cost models are 
  versioned, explicit inputs, ensuring all parameter changes are tracked via 
  version control diffs rather than hidden inline variables.
* Secure Runtime: Decouples the serving path from training dependencies and 
  pickled artifacts (such as SHAP explainers), eliminating arbitrary code execution 
  risks and minimizing the container footprint for edge deployment.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

API_VERSION = "v1"
SERVICE_VERSION = "2.0.0"
CONTRACT_VERSION = "2.0.0"

QualityTier = Literal["L", "M", "H"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PDM_", env_file=".env", extra="forbid", frozen=True
    )

    # ---- service ---------------------------------------------------------
    project_name: str = "Industrial PdM Service"
    environment: Literal["local", "dev", "staging", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"

    # ---- security --------------------------------------------------------
    # Empty in local only. `require_api_key` enforces that prod cannot boot
    # without one -- the previous service had no authentication at all.
    api_keys: str = ""
    rate_limit_per_minute: int = 600
    cors_allow_origins: str = ""

    # ---- model artifacts -------------------------------------------------
    # The rule engine and hazard model need NO artifacts: they are closed form.
    # An ONNX classifier is optional and only consulted for failure modes that
    # earned deployment in training (see manifest.json ship/no-ship gate).
    model_dir: str = "models"
    onnx_classifier: str = ""          # e.g. "clf_TWF.onnx"; empty = disabled
    onnx_contract_digest: str = ""     # must match the loaded feature contract
    onnx_parity_tolerance: float = 1e-5

    # ---- failure-mode specification (equipment envelope) -----------------
    # PROVENANCE: machine specification. NOT tuning knobs. A change here is a
    # spec change and requires sign-off.
    hdf_delta_t_k: float = 8.6
    hdf_speed_rpm: float = 1380.0
    pwf_min_w: float = 3500.0
    pwf_max_w: float = 9000.0
    osf_limit_l: float = 11_000.0
    osf_limit_m: float = 12_000.0
    osf_limit_h: float = 13_000.0
    tool_life_min_minutes: float = 200.0
    tool_life_max_minutes: float = 240.0

    # ---- decision policy -------------------------------------------------
    # The previous service escalated to IMMEDIATE_SHUTDOWN whenever the
    # predicted class was anything but Healthy, on top of a TWF threshold of
    # 0.12 whose measured precision was 0.06. That combination orders a line
    # stop on roughly 5% of healthy machines. Escalation is now driven by the
    # deterministic spec margins, which are exact, and by a hazard probability
    # with an explicit horizon.
    hazard_horizon_minutes: float = 10.0
    hazard_stop_probability: float = 0.60      # P(tool fails within horizon)
    hazard_schedule_probability: float = 0.20
    margin_warning_fraction: float = 0.80      # 80% of an envelope consumed
    margin_critical_fraction: float = 0.95

    # ---- cost model (for the reported expected-cost delta) ---------------
    cost_missed_failure: float = 12_000.0
    cost_false_alarm: float = 350.0
    cost_planned_stop: float = 900.0

    # ---- joint physics plausibility --------------------------------------
    # The per-channel bounds below catch a single impossible number. These
    # catch COMBINATIONS that are individually in range but jointly
    # impossible -- the signature of a drifting transducer rather than a
    # machine fault. Acting on those is the most common cause of alarm floods.
    plausibility_enabled: bool = True
    implausible_power_factor: float = 1.5     # x pwf_max_w
    implausible_delta_t_max_k: float = 20.0
    implausible_delta_t_min_k: float = 0.0    # process colder than air

    # Optional asset registry. Empty accepts any machine_id, which is fine
    # locally but lets a typo create a phantom machine in the audit log and
    # silently drop those alerts out of the work-order reconciliation.
    known_machine_ids: str = ""

    # ---- per-channel bounds; outside these it is a sensor fault ----------
    bound_temp_air_k: tuple[float, float] = (290.0, 315.0)
    bound_temp_process_k: tuple[float, float] = (295.0, 325.0)
    bound_speed_rpm: tuple[float, float] = (500.0, 4000.0)
    bound_torque_nm: tuple[float, float] = (0.0, 120.0)
    bound_tool_wear_min: tuple[float, float] = (0.0, 300.0)

    # ---- audit -----------------------------------------------------------
    # Every prediction is written here so alert precision can later be measured
    # against confirmed work orders. Without this you can never answer "is the
    # model actually useful", only "what did it score on a test set".
    prediction_log_path: str = "logs/predictions.jsonl"
    prediction_log_enabled: bool = True

    # ------------------------------------------------------------------
    @field_validator("tool_life_max_minutes")
    @classmethod
    def _life_window_ordered(cls, v, info):
        lo = info.data.get("tool_life_min_minutes")
        if lo is not None and v <= lo:
            raise ValueError("tool_life_max_minutes must exceed tool_life_min_minutes")
        return v

    @field_validator("pwf_max_w")
    @classmethod
    def _power_band_ordered(cls, v, info):
        lo = info.data.get("pwf_min_w")
        if lo is not None and v <= lo:
            raise ValueError("pwf_max_w must exceed pwf_min_w")
        return v

    @model_validator(mode="after")
    def _prod_requires_auth(self):
        if self.environment in ("staging", "prod") and not self.api_key_set:
            raise ValueError(
                "PDM_API_KEYS must be set outside local/dev: this endpoint can "
                "order a production line stop and must not be anonymous.")
        return self

    # ---- derived ---------------------------------------------------------
    @property
    def api_key_set(self) -> frozenset[str]:
        return frozenset(k.strip() for k in self.api_keys.split(",") if k.strip())

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def machine_registry(self) -> frozenset[str]:
        return frozenset(m.strip() for m in self.known_machine_ids.split(",") if m.strip())

    @property
    def osf_limits(self) -> dict[str, float]:
        return {"L": self.osf_limit_l, "M": self.osf_limit_m, "H": self.osf_limit_h}

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "temp_air_k": self.bound_temp_air_k,
            "temp_process_k": self.bound_temp_process_k,
            "speed_rpm": self.bound_speed_rpm,
            "torque_nm": self.bound_torque_nm,
            "tool_wear_min": self.bound_tool_wear_min,
        }

    def fingerprint(self) -> str:
        """Hash of every value that can change a decision. Emitted in each
        response and every audit record, so a prediction can always be traced
        back to the exact policy that produced it."""
        payload = {k: v for k, v in self.model_dump().items()
                   if k not in ("api_keys", "log_level", "prediction_log_path")}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
