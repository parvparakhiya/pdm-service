# Deployment Review — v1 service

A self-review of my own first deployment, written against the bar I would apply
to a service that can order a production line to stop. I wrote v1, found these
defects in it, and rebuilt it as v2 — which is what this repository contains.
Verdict first, then the findings, then what shipped instead.

---

## Verdict

**Blocked. Three defects make it wrong or non-functional; one of them silently
corrupts every prediction.**

The structure was sound — FastAPI, Pydantic schemas, a config module, ONNX
serving, a separate UI — and that is real progress over a notebook. But the
service was built on the artifacts the pipeline review said not to ship, and the
attempt to make those artifacts work at serving time introduced a worse bug than
any in the training code.

| | v1 | v2 |
|---|---|---|
| Stateless | ✗ shared 60-row buffer | ✓ pure function of one reading |
| Horizontally scalable | ✗ replicas disagree | ✓ |
| Boots without artifacts | ✗ crash-loop at import | ✓ rules need none |
| Readiness distinct from liveness | ✗ | ✓ |
| Authentication | ✗ none | ✓ API key, enforced in prod |
| Errors leak internals | ✗ `detail=str(e)` | ✓ stable codes |
| Serving image | 1.53 GB, unpickles at boot | 333 MB, no pickle |
| Predictions recorded | ✗ | ✓ audit JSONL |
| Tests | 0 | 45 domain/service + 15 HTTP |

---

## P0 — blocks deployment

### 1. One shared rolling buffer for every machine and every request

`services.py` holds a module-level `StatefulFeatureBuffer` — a single
`deque(maxlen=60)` — and appends **every incoming request** to it, then computes
`rolling(10/30/60)` statistics over that deque. Four failures come out of one
object:

* **All machines share it.** There is no `machine_id` anywhere in the API, so a
  reading from one CNC is averaged together with the last 60 requests from every
  other machine. The features fed to the model describe a fleet that does not
  exist.
* **Not idempotent.** The same payload posted twice returns different answers,
  because the first call mutated the state the second reads. Retries, replays
  and debugging all become unsound.
* **Not scalable or thread-safe.** Two replicas behind a load balancer hold
  different buffers and disagree on identical input. Within one replica, FastAPI
  runs `def` endpoints in a threadpool, and the read-then-build-DataFrame
  sequence is unlocked.
* **Total train/serve skew.** On the first request the buffer holds one row, so
  every `rolling_mean` equals the current value and every `rolling_std` is
  exactly `0.0` — values the model never saw in training, where the same columns
  were computed over 10,000 rows.

The root cause is that those rolling features were never legitimate. The training
data has no machine identity and no clock, so a 60-row window averaged 60
unrelated products; ablation showed removing all twelve changed macro-F1 by
**−0.002 — no measurable contribution**. **A buffer cannot fix a feature that
has no meaning.** v2 drops them, which is what makes the service stateless.

### 2. The feature contract does not resolve — the service cannot answer a request

```python
self.classifier_features = metadata.get('classifier_features', [])
self.rul_features        = metadata.get('rul_features', [])
```

`pipeline check.ipynb` writes the key `features`, not `classifier_features`, and
writes no `rul_features` at all. Both lists come back **empty**, so `build_vector`
returns a `(1, 0)` array and the ONNX session raises on the first request. The
paths disagree too: `config.py` points at `models/pdm_classifier.onnx` and
`models/metadata.pkl`, while the notebook saves `models/pdm_model.onnx` and
`models/pdm.metadata.pkl`.

This is the positional-contract finding from the pipeline review, now live: a
feature list inside a pickle, silently defaulting to `[]`.

### 3. The service escalates to shutdown on healthy machines

```python
if probs[1] > self.twf_threshold:   # 0.12, measured precision 0.06
    pred_class_idx = 1
...
if predicted_rul <= 15.0 or ... or failure_status != "Healthy":
    health_status, urgency = "CRITICAL", "IMMEDIATE_SHUTDOWN"
```

Any non-healthy class escalates to a line stop, and the TWF override forces class
1 for any reading above 0.12 probability. Reproduced on the full dataset, that
operating point raises **224 false alarms against 18 true catches — about 12:1**.
An alert that is wrong more than nine times in ten gets switched off, and then
the few that were real go unactioned too.

`predicted_rul <= 15.0` compounds it: that number comes from the regressor with
out-of-sample **R² = −0.104**.

---

## P1 — must fix before go-live

**4. `inference_service = UnifiedInferenceService()` runs at module import.**
A missing artifact raises before uvicorn binds, so the container crash-loops with
no endpoint to explain why. Load in a lifespan handler and report state on
`/ready`.

**5. `/health` is liveness pretending to be readiness.** It returns 200
unconditionally, so an orchestrator routes traffic to a pod whose model failed to
load.

**6. `detail=str(e)` leaks internals.** Container paths, library names and stack
context go to any caller who can POST. Return a stable code plus a request id;
send the detail to the log.

**7. No authentication, no rate limiting, no request ids, no logs, no metrics.**
An unauthenticated endpoint that can stop a production line. And with no
prediction log, alert precision can never be measured against confirmed work
orders — the one number the model card requires.

**8. `shap` + `lightgbm` + `scikit-learn` in the serving image to unpickle a
`TreeExplainer`.** This defeats the entire point of ONNX serving: several
hundred megabytes of dependencies, an artifact version-locked to the training
environment, and `joblib.load` as an arbitrary-code-execution surface at
startup.

**9. Softmax over probabilities.**

```python
if not np.isclose(np.sum(probs), 1.0, atol=1e-3):
    probs = exp_vals / np.sum(exp_vals)
```

If the ONNX output is not a probability distribution, that is a conversion bug.
Renormalising with a softmax turns it into plausible-looking numbers and hides
the defect. v2 raises instead.

**10. Docker.** `CMD uvicorn ... & streamlit ...` — shell form, so PID 1 is the
shell, uvicorn is an unreaped background job, and **if the API dies the container
stays healthy** because Streamlit is still running. SIGTERM never reaches
uvicorn, so `docker stop` cuts in-flight requests. Runs as root. No healthcheck,
no multi-stage build, no `.dockerignore` (so notebooks, CSVs and `.git` ship in
the image), and every dependency pinned with `>=`.

---

## P2 — correctness

**11. `product_type: str` with a silent fallback.** `OSF_LIMITS.get(tier, 12000.0)`
applies the *medium* threshold to any unrecognised tier. A typo produces a
confident, wrong answer.

**12. No sensor range validation.** Negative torque and 50,000 rpm are accepted
and scored.

**13. The UI's XAI advice is wrong and unsafe.** `if numeric_val > 0: "Too High
… Action: Decrease {name}"` reads a SHAP sign as a statement about the raw
reading. A positive SHAP value means the feature pushed the prediction toward
that class — not that the sensor is too high, and for several features decreasing
it makes things worse. Those SHAP values were also computed on the fabricated
buffer.

**14. UI control flow.** `if xai_data:` is dedented out of the `status_code == 200`
branch, so on any non-200 response `xai_data` is undefined and the operator gets
a `NameError` instead of the error message. The `else: st.error("API Error")`
branch also fires on *successful* healthy predictions, which legitimately have no
drivers. `st.columns(len(xai_data))` raises when empty. No timeout on
`requests.post`. API URL hardcoded to `127.0.0.1`.

**15. `Field(..., example=...)`** is deprecated in Pydantic v2; use
`json_schema_extra`. `all_class_probabilities: dict` and `root_cause_analysis: dict`
are untyped, so consumers guess.

---

## What shipped instead

The rule engine and hazard model are closed form, so **the service needs no model
artifacts to run**. That single fact removes the crash-loop, the pickle, the ML
dependency stack, and the contract-resolution bug at once — 1.53 GB down to
333 MB.

* `app/domain/spec.py` — the deterministic detector, plus envelope *margins*, so
  the console can warn before a limit is breached rather than only after.
* `app/domain/hazard.py` — closed-form tool-wear hazard with an explicit horizon,
  replacing `predicted_rul_minutes`.
* `app/domain/decision.py` — escalation from fired predicates and envelope
  pressure, every alert carrying its expected cost against doing nothing.
* `app/domain/features.py` — single-reading feature vector; the buffer is gone.
* `app/inference.py` — optional ONNX, refused unless its contract digest matches
  the code. Never unpickles. Fails soft to `/ready`.
* `app/observability.py` — JSON logs with request ids, Prometheus metrics, and
  the prediction audit trail that makes precision measurable.
* `docker/`, `docker-compose.yml`, `.github/workflows/ci.yml` — non-root
  multi-stage images, one process each, healthchecks, pinned deps, a CI job that
  fails if the feature contract digest drifts.

**Explainability improved by removing SHAP.** For a specification rule the
explanation is not an attribution estimate — it is the rule:

> Overstrain 13,200 Nm·min exceeds the 12,000 Nm·min limit for tier M (110% of
> capacity).

An operator can verify that by hand. A SHAP value cannot be verified by anyone.
