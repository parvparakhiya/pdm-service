# Runbook — Industrial PdM Service

On-call reference. Written to be read at 3am by someone who did not build it.

---

## 1. What this service does to the plant

It emits five outcomes. Only three are actionable:

| Severity | Urgency | Meaning | Who acts |
|---|---|---|---|
| `CRITICAL` | `STOP_MACHINE` | A specification limit is **measurably breached**, or tool-failure probability ≥ 60% within the horizon | Line supervisor, immediately |
| `DEGRADED` | `SCHEDULE_MAINTENANCE` | ≥95% of an operating envelope consumed, or ≥20% tool-failure probability | Maintenance planner, this shift |
| `WATCH` | `MONITOR` | 80–95% of an envelope consumed | Nobody. Trend only |
| `NOMINAL` | `NONE` | Inside spec on every mode | Nobody |
| `DATA_QUALITY` | `CHECK_INSTRUMENTATION` | The reading was not physically possible; **the machine was not assessed** | Instrumentation tech, before the machine is trusted |

`DATA_QUALITY` is not a rung on the same ladder as the other four — it answers a
different question. It means the channels contradicted each other, so no verdict
was computed. **Do not read it as "all clear".** The response body still lists
what each failure-mode predicate would have said, and the advisory states plainly
whether the machine must be stopped if the instruments check out.

**A `CRITICAL` from a specification rule is not a prediction.** The machine is
outside its documented operating envelope right now. Treat it as an instrument
reading, not a model output. The `evidence` field states the exact numbers and
can be checked by hand.

---

## 2. Service level objectives

| SLO | Target | Alert at |
|---|---|---|
| Availability (`/ready` 200) | 99.5% monthly | 2 consecutive failures |
| p99 scoring latency | < 25 ms | p99 > 100 ms for 5 min |
| Error rate (5xx) | < 0.1% | > 1% for 5 min |
| False-alarm ratio, `STOP_MACHINE` | ≤ 3 per true catch | > 5 over a rolling 7 days |

The last one is the important one and it is **not** measurable from the service
alone — it comes from the weekly reconciliation in §6. An alert stream noisier
than about 3:1 gets ignored by the floor, at which point the true positives stop
being actioned too. If that ratio drifts, the correct response is to raise the
thresholds, not to ask the floor to try harder.

### Prometheus alerts

```yaml
- alert: PdmNotReady
  expr: pdm_ready == 0
  for: 2m
  annotations:
    summary: "PdM service not ready"
    runbook: "RUNBOOK.md#4-service-will-not-start"

- alert: PdmHighLatency
  expr: histogram_quantile(0.99, rate(pdm_request_seconds_bucket[5m])) > 0.1
  for: 5m

- alert: PdmErrorRate
  expr: rate(pdm_requests_total{status=~"5.."}[5m]) / rate(pdm_requests_total[5m]) > 0.01
  for: 5m

# Escalation storm: something upstream is wrong (bad sensor feed, wrong tier
# mapping) far more often than every machine failing at once.
- alert: PdmStopOrderStorm
  expr: rate(pdm_decisions_total{urgency="STOP_MACHINE"}[15m]) > 0.05
  for: 10m
  annotations:
    summary: "Stop orders above 3/min — suspect the telemetry feed, not the fleet"
    runbook: "RUNBOOK.md#5-too-many-stop-orders"
```

---

## 3. Normal operations

```bash
make up                     # start
docker compose ps           # both services healthy?
curl -s localhost:8000/ready | jq
docker compose logs -f api | jq -r '"\(.ts) \(.level) \(.message)"'
```

Every log line is JSON and carries `request_id`. To trace one request end to end:

```bash
docker compose logs api | jq 'select(.request_id=="<id>")'
```

`policy_fingerprint` appears on every response and every audit record. If two
predictions disagree, compare fingerprints first — a different fingerprint means
a different policy, not a bug.

---

## 4. Service will not start

Check `/ready` and the startup log. In order of likelihood:

**`PDM_API_KEYS must be set outside local/dev`** — intended. This endpoint can
stop a production line and will not run anonymously. Set the key.

**`feature contract mismatch: artifact digest X != service digest Y`** — a
deployed ONNX model was trained on a different feature set or column order than
this build expects. **Do not bypass this.** It is the check that prevents
silently wrong predictions. Either roll back the image to the one matching the
artifact, or re-export the model and regenerate its `.contract.json`.

**`<file> not found`** for an ONNX artifact — if no model has passed the
ship/no-ship gate, clear `PDM_ONNX_CLASSIFIER`. The rules and hazard need no
artifacts and the service will be fully functional without one.

**A mistyped setting** — `extra="forbid"` means `PDM_HAZRD_STOP_PROBABILITY`
fails at boot rather than silently using the default. Read the error; it names
the field.

> The service never fails because a *model* is missing. If it will not start, it
> is configuration.

---

## 5. Too many stop orders

Almost always upstream, not the fleet. Work in this order:

1. **Look at the audit log**, not the dashboard:
   ```bash
   docker compose exec api sh -c \
     'tail -500 /app/logs/predictions.jsonl' | jq -r \
     'select(.urgency=="STOP_MACHINE") | [.machine_id,(.triggered_by|join(";"))] | @tsv' \
     | sort | uniq -c | sort -rn | head
   ```
2. **One machine dominating** → likely a real fault, or a failed sensor on that
   machine. Check the raw `input` values in the same records for something
   physically implausible that still passed validation.
3. **One failure mode dominating across machines** → suspect a unit or mapping
   error in the feed. The classic is `product_type` defaulting to the wrong tier,
   which shifts the OSF limit by up to 2,000 Nm·min.
4. **All modes, all machines** → the telemetry feed changed. Compare the `input`
   distribution in the audit log against last week.
5. **Only then** consider the thresholds. Changing `PDM_HAZARD_STOP_PROBABILITY`
   or `PDM_MARGIN_CRITICAL_FRACTION` is a policy change: it alters the
   fingerprint, needs review, and must be recorded in `docs/DECISIONS.md`. It is
   not a hotfix.

**Never** silence an alert by widening a specification constant
(`PDM_HDF_DELTA_T_K`, `PDM_PWF_*`, `PDM_OSF_LIMIT_*`). Those are the machine's
documented envelope. Changing one is an engineering change requiring sign-off
from whoever owns the equipment spec.

---

## 5a. A machine reports DATA_QUALITY

The joint-physics checks caught a reading whose channels contradict each other.
Each value was inside its own limits; the combination was not possible.

```bash
docker compose exec api sh -c 'tail -500 /app/logs/predictions.jsonl' | jq -r \
  'select(.data_quality | length > 0) | [.machine_id,(.data_quality|join(";"))] | @tsv' \
  | sort | uniq -c | sort -rn
```

| Code | Meaning | Check |
|---|---|---|
| `power_above_rating` | torque × speed implies more than 1.5× rated shaft power | Torque transducer calibration and zero; speed/encoder scaling |
| `thermal_gradient_inverted` | process colder than ambient | Probes swapped, or one has failed open |
| `thermal_gradient_excessive` | gradient beyond the plausible band | Drifting temperature channel; coolant flow sensor |
| `tool_wear_beyond_specified_life` | *advisory* — wear past the specified maximum | Wear counter not reset at the last tool change |

**One machine repeating the same code** is almost always that machine's
instrument. **Many machines at once** means a scaling or unit change upstream —
check whether the historian or gateway configuration was modified.

Do not widen `PDM_IMPLAUSIBLE_*` to make these go away. Those limits are what
stop an instrument fault from being reported as a machine fault.

## 6. Weekly: is it actually useful?

Precision measured against the label is not precision measured against reality.
Reconcile the audit log with confirmed work orders:

```bash
# stop orders raised in the last 7 days
jq -r 'select(.urgency=="STOP_MACHINE") | [.ts,.machine_id,.request_id] | @tsv' \
  /app/logs/predictions.jsonl > /tmp/alerts.tsv
# join against the CMMS export on (machine_id, ts window) and compute:
#   true catches   = alerts with a confirmed failure or replacement within 1h
#   false alarms   = alerts with no corresponding work order
#   missed         = work orders with no alert in the preceding hour
```

Report `false_alarms / true_catches` against the ≤3 SLO. If it exceeds 5, raise
the thresholds that week — do not wait for a quarterly review.

Also watch **rule-versus-model disagreement** if an ONNX model is deployed. A
rising rate means the equipment envelope has moved and the specification
constants need a human, not a retrain.

---

## 7. Rollback

Images are immutable and tagged. Nothing in the request path is stateful, so
rollback is a redeploy — no draining, no migration, no cache to warm.

```bash
docker compose down
docker compose up -d --no-build   # pin the previous tag in docker-compose.yml
curl -s localhost:8000/ready | jq -r .policy_fingerprint
```

Confirm the fingerprint matches the version you intended. The audit log is
append-only and survives; predictions from both versions are distinguishable by
their fingerprint.

To disable a misbehaving learned model without a rollback, clear
`PDM_ONNX_CLASSIFIER` and restart: scoring falls back to the closed-form hazard,
which is always available.

---

## 8. Escalation

| Symptom | Owner |
|---|---|
| Service down, config error, latency | Platform on-call |
| Stop-order storm, threshold change | ML on-call + maintenance planner |
| Specification constant looks wrong | Equipment engineering (sign-off required) |
| Alert ratio SLO breached 2 weeks running | ML lead — the model needs re-scoping, not tuning |

---

## 9. Known limitations (state these when asked)

* **No lead time.** Specification predicates fire when the limit is breached.
  Early warning comes only from envelope margins, and only for OSF and PWF.
* **Tool-wear risk is a base rate, not a per-machine forecast.** Mutual
  information between the sensor channels and tool-wear failure is effectively
  zero — the sensors carry no information about when the tool-life draw lands.
  Two machines at the same wear get the same number, correctly. Reproduce with
  `pdm.hazard.irreducibility_report` in the `pdm-audit` repository.
* **No RUL.** Removed deliberately; see `docs/DECISIONS.md`.
* **The rate limiter is per process.** With N replicas the effective limit is
  N × the configured value. Enforce the real limit at the gateway.
