# Architecture decisions

Ten records. Each one is a call someone will question later; the reasoning is
here so the answer does not depend on who is in the room.

All ten are implemented and covered by tests in this repository. Where a record
refers to **v1**, it means the first deployment of this service, whose failures
are documented in [../DEPLOYMENT_REVIEW.md](../DEPLOYMENT_REVIEW.md). Where it
refers to the **training pipeline**, that is the companion repository
`pdm-audit`, which produced the evidence quoted here.

| | Decision |
|---|---|
| ADR-001 | Specification rules are the primary detector, not a model |
| ADR-002 | No learned RUL model is deployed |
| ADR-003 | The service is stateless; there is no rolling-feature buffer |
| ADR-004 | Explanations come from the rules, not from SHAP |
| ADR-005 | Escalation is cost-based, with a hard alert budget |
| ADR-006 | The feature contract is named, digested and enforced at startup |
| ADR-007 | API and UI are separate containers |
| ADR-008 | Sensor validation precedes condition monitoring |
| ADR-009 | Operator wording is not dataset wording |
| ADR-010 | `machine_id` is validated against an asset registry |

---

## ADR-001 — Specification rules are the primary detector, not a model

**Status:** accepted — implemented

**Context.** HDF, PWF and OSF are defined by the machine specification as
closed-form predicates over the sensor readings. The v1 pipeline engineered
`temp_delta_k`, `power_watts` and `overstrain_index` — the decision variables of
those very predicates — and trained a tuned LightGBM classifier on them.

**Decision.** Ship the predicates. Consult a model only where one passes a
ship/no-ship gate against them.

**Evidence.** Three predicates score macro-F1 **0.796**; the tuned 50-candidate
LightGBM pipeline scores **0.766**. Against the published labels the predicates
agree exactly: HDF 1.000, PWF 1.000, OSF 1.000. The training pipeline's gate
returned BASELINE for all four failure modes, so no learned model was deployed.

**Consequences.** No artifacts on the primary path, so no crash-loop, no pickle,
no version lock, and a **333 MB** serving image where v1 of the same application
built to **1.53 GB** — a 4.6x reduction, measured on both images. Latency is
microseconds. Every alert is verifiable by hand. The cost is that the rules
cannot generalise beyond the documented envelope — if the real machine deviates
from its spec, only a model would notice, which is what the rule-versus-model
disagreement metric in the runbook is for.

**Where it lives.** `app/domain/spec.py`.

---

## ADR-002 — No learned RUL model is deployed

**Status:** accepted — implemented

**Context.** v1 served `rul_model.onnx` as `predicted_rul_minutes` and used it to
order line stops at ≤ 15 minutes.

**Decision.** Delete it. Report a tool-wear hazard with an explicit horizon
instead: *P(failure within 10 minutes)*, plus expected remaining life, both
closed form.

**Evidence.** The training target was `clip(limit(quality) − tool_wear, 0, 120)` —
one subtraction. Its reported R² = 0.845 came from neighbouring rows; under
contiguous-block cross-validation the same model scores **−0.104**, worse than
predicting the mean. Tool life is drawn on [200, 240] min independently of every
sensor, so given survival to wear *w*, remaining life is exactly
`Uniform(max(w, 200), 240)` — and no model can beat a closed form when the
randomness is in the process rather than in our ignorance.

**Consequences.** The API no longer returns a number that looks like a forecast
and is not one. Stakeholders must be told this is detection with margin-based
early warning, not lead-time prediction. The upgrade path is
`empirical_hazard_from_lifetimes`, which needs run-to-failure records the current
historian does not produce.

**Where it lives.** `app/domain/hazard.py`.

---

## ADR-003 — The service is stateless; there is no rolling-feature buffer

**Status:** accepted — implemented

**Context.** v1 kept a module-level `deque(maxlen=60)` shared by every machine
and every request, and computed rolling statistics over it.

**Decision.** No server-side history. Scoring is a pure function of one reading.

**Rationale.** The buffer produced wrong answers four different ways —
cross-machine contamination, non-idempotence, replica divergence, and
first-request train/serve skew where every rolling std was exactly 0. But the
deeper point is that the rolling features were never meaningful: the training
data has no machine identity and no clock, so a 60-row window averaged 60
unrelated products, and ablating all twelve changed macro-F1 by **−0.002** — no
measurable contribution. A better buffer cannot rescue a feature with no
referent.

**Consequences.** Horizontal scaling is free, retries are safe, batch equals
sequential, and rollback needs no draining. If genuine per-machine history
arrives it belongs in a feature store keyed by `machine_id` with point-in-time
correctness — not in process memory.

**Where it lives.** `app/domain/features.py`, `app/service.py`.

---

## ADR-004 — Explanations come from the rules, not from SHAP

**Status:** accepted — implemented

**Context.** v1 unpickled a live SHAP `TreeExplainer` at startup and returned the
top three attribution values as "root cause analysis". The console rendered
positive values as *"Too High — Action: Decrease {feature}"*.

**Decision.** Return the predicate and its numbers. Drop SHAP from the serving
path entirely.

**Rationale.** For a specification rule the explanation is not an estimate, it is
the rule:

> Overstrain 13,200 Nm·min exceeds the 12,000 Nm·min limit for tier M (110% of
> capacity).

An operator can check that against the machine. A SHAP value cannot be checked by
anyone. The v1 rendering was also unsound: a positive SHAP value means the feature
pushed the prediction toward that class, not that the sensor reading is too high —
and those values were computed over the fabricated buffer from ADR-003.

**Consequences.** `shap`, `lightgbm` and `scikit-learn` leave the serving image,
and `joblib.load` stops being an arbitrary-code-execution surface at boot.
Offline SHAP analysis on held-out data remains useful and belongs in `pdm-audit`,
never on the serving path.

**Where it lives.** `app/domain/spec.py` (the `evidence` field);
`requirements/api.txt` documents what was removed and why.

---

## ADR-005 — Escalation is cost-based, with a hard alert budget

**Status:** accepted — implemented

**Context.** v1 escalated to `IMMEDIATE_SHUTDOWN` whenever the predicted class
was not Healthy, on top of a TWF threshold of 0.12 whose reported precision was
0.06 — roughly one stop order per twenty healthy machines.

**Decision.** Escalate from (a) a predicate that has actually fired, or (b) a
hazard probability over a stated horizon, or (c) measured envelope pressure.
Every alert reports its expected cost against doing nothing, and the
false-alarms-per-catch ratio is an SLO with a hard ceiling of 3.

**Rationale.** An alert that is wrong more than nine times in ten gets switched
off, and then the few that were real go unactioned too. Alert precision is a
safety property here, not a UX preference.

**Consequences.** Thresholds are arguable with numbers in a review rather than
adjustable by feel in a hotfix. Changing one alters the policy fingerprint
recorded on every prediction, so the change is visible in the audit trail. The
cost figures in `PDM_COST_*` are placeholders until finance and maintenance
agree real ones; every threshold below is justified against them.

**Where it lives.** `app/domain/decision.py`, `app/config.py`.

---

## ADR-006 — The feature contract is named, digested and enforced at startup

**Status:** accepted — implemented

**Context.** v1's contract was `FloatTensorType([None, 21])` plus a feature list
inside a pickle — and the key the service read (`classifier_features`) was not
the key the notebook wrote (`features`), so the list came back empty and the
session was invoked with a zero-width tensor.

**Decision.** `FEATURE_ORDER` in code is the contract. Its SHA-256 prefix is
`6c41df1031b28f72`. An ONNX artifact must ship a `.contract.json` whose digest
matches, or it is refused at startup and reported on `/ready`. CI fails if the
digest changes without the artifacts being re-exported.

**Consequences.** A reordered column is a loud startup failure instead of silently
wrong predictions. Changing the feature set is a deliberate two-part change: code
plus re-exported artifact.

**Where it lives.** `app/domain/features.py`, `app/inference.py`, and the
"Feature contract digest is unchanged" step in `.github/workflows/ci.yml`.

---

## ADR-007 — API and UI are separate containers

**Status:** accepted — implemented

**Context.** v1 ran `uvicorn ... & streamlit ...` from a single shell `CMD`.

**Decision.** One process per container, exec-form entrypoint under `tini`,
non-root, with its own healthcheck.

**Rationale.** In the v1 arrangement PID 1 was the shell, uvicorn was an unreaped
background job, and **if the API died the container still reported healthy**
because Streamlit was alive — an outage no orchestrator could detect. SIGTERM
also never reached uvicorn, so `docker stop` cut in-flight requests instead of
draining them.

**Consequences.** The two scale and roll back independently, and API failure is
visible. The UI now reaches the API by service name, so `PDM_API_BASE` must be
set.

**Where it lives.** `docker/Dockerfile.api`, `docker/Dockerfile.ui`,
`docker-compose.yml`. CI asserts the image does not run as root.

---

## ADR-008 — Sensor validation precedes condition monitoring

**Status:** accepted — implemented

**Context.** Input validation was per-channel: torque ≤ 120 Nm, process
temperature ≤ 325 K, each checked alone. A reading of 100 Nm at 1408 rpm with a
26.8 K thermal gradient passed every bound and was scored `CRITICAL /
STOP_MACHINE`. That reading implies 14,745 W of shaft power — 164% of the
9,000 W rating — while the coolant gradient has nearly tripled. No spindle does
that. It is a transducer fault wearing the costume of a machine fault.

**Decision.** Add a joint-physics layer ahead of the failure-mode logic. When a
reading is not self-consistent, return `DATA_QUALITY / CHECK_INSTRUMENTATION`
naming the channel to check, instead of a machine verdict.

**Rationale.** The most common cause of alarm floods in real plants is
instrumentation failing, not machines failing. An operator told to stop
production because a torque sensor drifted learns to ignore the system — the
same alert-fatigue failure ADR-005 exists to prevent, arriving through a
different door. Per-channel bounds catch one impossible number; only a joint
check catches an impossible combination. Sensor validation is a standard layer
upstream of condition monitoring (EEMUA 191, ISA-18.2).

**What this deliberately does not do.** It does not suppress the findings. Every
predicate is still evaluated and returned, and the advisory states explicitly
whether the machine must be stopped if the instruments check out. Silently
hiding a possible breach would trade one failure mode for a worse one.

**Consequences.** A fourth severity that is orthogonal to the other three, and
one advisory code (`tool_wear_beyond_specified_life`) that is deliberately
non-blocking — a stale wear counter is likely, but a worn tool needs replacing
either way. The thresholds (`implausible_power_factor`, the ΔT band) are
policy: they sit in config, change the policy fingerprint, and must not be
widened to silence an alarm.

**Where it lives.** `app/domain/plausibility.py`, `app/service.py`, and the
`PDM_IMPLAUSIBLE_*` settings in `app/config.py`.

---

## ADR-009 — Operator wording is not dataset wording

**Status:** accepted — implemented

**Context.** The dataset names a failure mode "Power Failure" (PWF), and that
string reached the operator console: *"Specification limit exceeded for power
failure."*

**Decision.** Internal codes stay (`PWF`); operator-facing text says *"shaft
power outside its envelope"*.

**Rationale.** On a shop floor, "power failure" means the electricity went out —
a completely different response from an overloaded spindle. Inheriting a
dataset's label into a safety-relevant advisory is how the wrong action gets
taken.

**Where it lives.** `app/domain/decision.py` (`MODE_NAMES`). Enforced by
`test_no_ambiguous_power_failure_wording`, which fails if the phrase reappears
anywhere in the operator-facing text.

---

## ADR-010 — `machine_id` is validated against an asset registry

**Status:** accepted — implemented

**Context.** `machine_id` was free text, so a keyboard-mash id scored
successfully and was written to the audit log as though it were a real asset.

**Decision.** When `PDM_KNOWN_MACHINE_IDS` is configured, an unlisted id is
rejected with 422. An empty registry accepts anything, which is intended for
local development only — staging and production must set it.

**Rationale.** A typo creates a phantom asset in the audit log, and those alerts
then vanish from the weekly work-order reconciliation — quietly corrupting the
one measurement that says whether the system is useful.

**Where it lives.** `app/config.py` (`machine_registry`), `app/service.py`.
Enforced by `test_unknown_machine_is_rejected_when_registry_configured`.
