# models/

**No model artifacts ship here, and that is the correct state.**

The specification rules and the tool-wear hazard are closed form — they need no
artifacts. The service starts, passes `/ready` and scores correctly with nothing
here. That is why v2 cannot crash-loop on a missing file the way v1 did.

This file also keeps the directory present in git, because `docker/Dockerfile.api`
copies `models/` into the image. Do not delete it.

## Why nothing is here

An ONNX classifier belongs here **only** when the training pipeline's
ship/no-ship gate approved one for its failure mode — meaning the model beat the
cheap baseline on held-out expected cost, not on macro-F1.

On the AI4I 2020 dataset that gate returned **BASELINE for all four failure
modes**. HDF, PWF and OSF are closed-form predicates over the sensors, so the
rules match the labels exactly and a model cannot improve on them. TWF is an
independent draw on [200, 240] minutes, so the analytic hazard beats every
learned candidate.

The evidence is the `decision` field on each entry of the `results` array in
[`pdm-audit/artifacts/manifest.json`](https://github.com/parvparakhiya/pdm-audit).

The absence of a model here is a finding, not an omission.

## Adding an approved model

Two files, always as a pair:

```
models/clf_<MODE>.onnx
models/clf_<MODE>.onnx.contract.json
```

The contract mirrors `artifacts/serving_contract.json`, plus the two fields that
are specific to the deployed model:

```json
{
  "feature_names": ["temp_air_k", "temp_process_k", "..."],
  "dtype": "float32",
  "schema_version": "2.0.0",
  "digest": "6c41df1031b28f72",
  "mode": "TWF",
  "threshold": 0.42
}
```

`feature_names` must be the full 16 entries in the order given by
`app.domain.features.FEATURE_ORDER` — `artifacts/serving_contract.json` in this
repo is the current list. `threshold` above is illustrative; use the operating
point the gate selected.

`digest` must equal `contract_digest()` for the build you are deploying. From
the repository root:

```bash
python -c "from app.domain.features import contract_digest; print(contract_digest())"
```

If the digest does not match, the service refuses the model at startup and says
so on `/ready`. **Do not work around that check** — it is what stops a reordered
column from producing silently wrong predictions, and CI fails on the same
comparison.

Then set `PDM_ONNX_CLASSIFIER=clf_<MODE>.onnx` and build with
`--build-arg WITH_ONNX=true`, which installs `requirements/api-onnx.txt`
instead of `requirements/api.txt`.

To disable a misbehaving model without a rollback, clear `PDM_ONNX_CLASSIFIER`
and restart. Scoring falls back to the rules and the closed-form hazard, which
are always available.

## Never put a pickle here

v1 loaded `metadata.pkl` containing a live SHAP `TreeExplainer`. That forced
lightgbm, shap and scikit-learn into the serving image and made startup an
arbitrary-code-execution surface across a trust boundary. Metadata is JSON.
Explanations come from the rules — see `docs/DECISIONS.md`, ADR-004.
