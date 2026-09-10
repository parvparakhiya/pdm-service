# Industrial PdM Service — v2

Failure-mode detection for CNC machines. Stateless, no model artifacts required,
non-root containers, 60 tests.

```
POST /api/v1/score      score one reading
POST /api/v1/score/batch historian backfill
GET  /health            liveness
GET  /ready             readiness (model state, contract digest, policy hash)
GET  /metrics           Prometheus
```

## What it does, and what it does not

**Does:** exact detection of the three specified failure modes — heat
dissipation, power envelope, overstrain — plus a closed-form tool-wear hazard
over an explicit horizon, and early warning through envelope margins. Ahead of
all of that, a joint-physics layer separates *instrument faults* from *machine
faults*, so a drifting transducer raises `CHECK_INSTRUMENTATION` rather than
stopping a production line.

**Does not:** predict remaining useful life. No learned RUL model is deployed;
the candidate scored **R² = −0.104** out of sample. The specification predicates
fire at the instant a limit is breached, so this is a *detector* with margin-based
early warning, not a lead-time forecast. Do not describe it to stakeholders as
predictive maintenance. See [docs/DECISIONS.md](docs/DECISIONS.md).

## Quick start

```bash
cp .env.example .env          # set PDM_API_KEYS and PDM_UI_API_KEY
make build && make up
make smoke
```

API on `127.0.0.1:8000`, operator console on `127.0.0.1:8501`.

Or run the published image without cloning anything:

```bash
docker run -p 8000:8000 -e PDM_ENVIRONMENT=dev -e PDM_API_KEYS=your-key \
  parvparakhiya/pdm-api:latest
```

```bash
curl -X POST localhost:8000/api/v1/score \
  -H 'content-type: application/json' -H 'x-api-key: YOUR_KEY' \
  -d '{"machine_id":"CNC-014","product_type":"M","temp_air_k":298.2,
       "temp_process_k":308.7,"speed_rpm":1408,"torque_nm":46.3,
       "tool_wear_min":115}'
```

Abridged — the real response returns a finding for all four modes, plus
`margins`, `data_quality`, `request_id` and `latency_ms`:

```json
{
  "decision": {
    "severity": "NOMINAL",
    "urgency": "NONE",
    "recommended_action": "Operating within specification on every monitored failure mode.",
    "triggered_by": [],
    "expected_cost_delta": 0.0
  },
  "tool_life": {
    "estimator": "analytic_uniform_tool_life",
    "horizon_minutes": 10.0,
    "p_failure_within_horizon": 0.0,
    "expected_remaining_minutes": 105.0
  },
  "findings": [{"mode": "OSF", "detected": false, "source": "specification_rule",
                "confidence": 1.0, "envelope_used": 0.4437,
                "evidence": "Overstrain at 44% of the 12,000 Nm·min limit for tier M."}],
  "policy_fingerprint": "<16 hex chars>"
}
```

`policy_fingerprint` is a hash of every setting that can change a decision, so
yours will differ from anyone else's. Two predictions with the same fingerprint
were produced under the same policy; two with different fingerprints were not.
Read it from `/ready`.

## Design in one paragraph

Three of the four failure modes are defined by the machine specification as
closed-form predicates over the sensors, so they are implemented as predicates —
exact, auditable, microseconds, and measurably better than the trained
classifier (macro-F1 0.796 vs 0.766). Tool wear is genuinely stochastic and is
handled as a hazard with a stated horizon. A learned model is consulted only for
a mode that passed the training pipeline's ship/no-ship gate, and is refused at
startup unless its feature-contract digest matches the code — on this dataset the
gate approved none of the four, so nothing learned is deployed. Because the
primary path needs no artifacts, the service starts and serves correctly with an
empty `models/` directory.

## Layout

```
app/
  config.py         validated settings; prod refuses to boot without API keys
  schemas.py        strict request/response contract, versioned
  service.py        all scoring logic — testable without a web server
  main.py           routing, auth, rate limit, error handling. Nothing else.
  inference.py      optional ONNX, contract-gated, never unpickles
  observability.py  JSON logs, metrics, prediction audit, token bucket
  domain/
    spec.py         the deterministic detector + envelope margins
    hazard.py       closed-form tool-wear hazard
    features.py     single-reading feature vector (and why there is no buffer)
    decision.py     escalation policy under an explicit cost model
ui/app.py           operator console
tests/              45 domain/service tests + 15 HTTP tests
docker/             non-root multi-stage images, one process each
```

## Operating it

* [RUNBOOK.md](RUNBOOK.md) — alerts, SLOs, failure modes, rollback.
* [DEPLOYMENT_REVIEW.md](DEPLOYMENT_REVIEW.md) — what was wrong with v1.
* [docs/DECISIONS.md](docs/DECISIONS.md) — the ten calls made, and why.
* [models/README.md](models/README.md) — why that directory is empty.

## Where the evidence comes from

Every measured figure quoted here is produced by
[**pdm-audit**](https://github.com/parvparakhiya/pdm-audit), the training and
leakage-audit repository for the same problem. It holds the ship/no-ship gate
that rejected the learned models, the ablations behind the rolling-feature and
RUL findings, and the manifest those numbers are read from.

The data is the public [AI4I 2020 predictive maintenance
dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
Its failure labels are generated by formula rather than observed on a machine,
which is why the specification rules reproduce them exactly and why the honest
conclusion here was to ship rules rather than a model.

## Before production

Four things are genuinely outstanding, and none are code:

1. **Replace the cost model.** `PDM_COST_*` are placeholders. Every escalation
   threshold is justified against them, so they need real figures from finance
   and maintenance.
2. **Wire the audit log to your lake** and build the weekly reconciliation
   against confirmed work orders. Until that exists you cannot answer whether
   the alerts are useful.
3. **Terminate TLS and authenticate at the gateway.** The API key here is a
   backstop, not an identity system.
4. **Instrument `machine_id`, `timestamp` and run-to-failure events in the
   historian.** That is the only thing that unlocks real lead-time prediction —
   more modelling on the current data will not.
