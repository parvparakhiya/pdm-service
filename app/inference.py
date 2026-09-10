"""Optional ONNX model registry with cryptographically bound feature contracts.

This module manages conditional artifact loading for secondary ML models. 
Because the deterministic rule engine and analytic hazard models operate entirely 
without external weights, learned artifacts are strictly optional and loaded only 
if they successfully clear the training pipeline's deployment gate.

Architectural guarantees enforced by this registry:

*   STRICT CONTRACT VERIFICATION: Replaces brittle positional arrays with named 
    feature contracts and cryptographic hash digests. Any mismatch between the 
    artifact's expected schema and the runtime environment triggers an immediate 
    startup failure.
*   SECURE SERIALIZATION: Metadata is managed exclusively via JSON. By decoupling 
    model weights from training-time dependencies and pickled objects, the serving 
    path eliminates arbitrary code execution risks and minimizes the container footprint.
*   RESILIENT LIFECYCLE MANAGEMENT (FAIL-SOFT): Initialization errors and missing 
    artifacts do not trigger import-time crash-loops. Faults are handled gracefully, 
    allowing the application to bind network ports and report status via health 
    and readiness probes (e.g., `/ready`).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any
from pathlib import Path
from typing import Literal

from .config import Settings
from .domain.features import FEATURE_ORDER, contract_digest

log = logging.getLogger("pdm.inference")

LoadState = Literal["loaded", "disabled", "failed"]


@dataclass
class ClassifierHandle:
    # onnxruntime is an optional dependency, so the session cannot be typed
    # here without importing it unconditionally.
    session: Any
    input_name: str
    mode: str
    threshold: float
    digest: str


class ModelRegistry:
    """Holds whatever optional artifacts loaded successfully."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.state: LoadState = "disabled"
        self.detail: str = "no ONNX classifier configured; rules + hazard only"
        self.classifier: ClassifierHandle | None = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        s = self.settings
        if not s.onnx_classifier:
            log.info("onnx classifier disabled by configuration")
            return

        path = Path(s.model_dir) / s.onnx_classifier
        meta_path = path.with_suffix(path.suffix + ".contract.json")
        try:
            if not path.exists():
                raise FileNotFoundError(f"{path} not found")
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"{meta_path} not found: an ONNX artifact without a feature "
                    "contract cannot be verified and will not be loaded")

            meta = json.loads(meta_path.read_text())
            expected = contract_digest()
            if meta.get("digest") != expected:
                raise ValueError(
                    f"feature contract mismatch: artifact digest {meta.get('digest')!r} "
                    f"!= service digest {expected!r}. The model was trained on a "
                    f"different feature set or a different column order. Refusing to load.")
            if list(meta.get("feature_names", [])) != list(FEATURE_ORDER):
                raise ValueError("feature names differ from FEATURE_ORDER despite "
                                 "matching digest; artifact is inconsistent")

            import onnxruntime as ort           # imported only when actually needed
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            shape = sess.get_inputs()[0].shape
            width = shape[-1]
            if isinstance(width, int) and width != len(FEATURE_ORDER):
                raise ValueError(f"model expects width {width}, contract has "
                                 f"{len(FEATURE_ORDER)}")

            self.classifier = ClassifierHandle(
                session=sess, input_name=sess.get_inputs()[0].name,
                mode=str(meta.get("mode", "TWF")),
                threshold=float(meta.get("threshold", 0.5)),
                digest=expected)
            self.state = "loaded"
            self.detail = f"onnx classifier for {self.classifier.mode} loaded from {path.name}"
            log.info("onnx classifier loaded", extra={"path": str(path), "digest": expected})

        except Exception as exc:                              # noqa: BLE001
            self.state = "failed"
            self.detail = f"{type(exc).__name__}: {exc}"
            log.error("onnx classifier failed to load: %s", self.detail)

    # ------------------------------------------------------------------
    def predict_proba(self, vector: list[float]) -> float | None:
        """Positive-class probability, or None when no model is deployed."""
        if self.state != "loaded" or self.classifier is None:
            return None
        import numpy as np

        h = self.classifier
        arr = np.asarray([vector], dtype=np.float32)
        out = h.session.run(None, {h.input_name: arr})

        proba = out[-1]
        '''zipmap=False is set at export time, so this is a plain array. Handle
        the mapped form anyway rather than assuming, and NEVER "fix" a
        non-normalised output by softmaxing probabilities the way the
        previous service did -- that silently converts a conversion bug into
        plausible-looking numbers.'''
        if isinstance(proba, list) and proba and isinstance(proba[0], dict):
            row = proba[0]
            return float(row.get(1, 0.0))
        arr_out = np.asarray(proba, dtype=np.float64)
        if arr_out.ndim == 2 and arr_out.shape[1] >= 2:
            arr_row = arr_out[0]
            total = float(arr_row.sum())
            if not (0.99 <= total <= 1.01):
                raise ValueError(
                    f"ONNX output does not sum to 1 (sum={total:.4f}); the export "
                    "is producing scores, not probabilities. Fix the conversion.")
            return float(arr_row[1])
        return float(np.ravel(arr_out)[0])

    # ------------------------------------------------------------------
    def readiness(self) -> dict:
        return {"onnx_classifier": self.state, "detail": self.detail,
                "contract_digest": contract_digest()}
