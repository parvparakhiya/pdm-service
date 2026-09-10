"""HTTP layer. Routing, auth, serialisation -- and nothing else.

Every decision rule lives in app/service.py and app/domain/, so this file stays
small enough to read in one sitting and the logic stays testable without a web
server.

Structural fixes over the previous main.py:

*   The service object is built in a lifespan handler, not at module import.
    `inference_service = UnifiedInferenceService()` at import time meant a
    missing artifact killed the process before uvicorn could bind, producing a
    crash-loop with no endpoint to explain it.
*   /health is liveness only; /ready reports whether the service can actually
    return a correct answer. The old /health returned 200 unconditionally, so an
    orchestrator would route traffic to a pod with a broken model.
*   Errors return a stable code and a request id. The old handler did
    `detail=str(e)`, leaking container paths and library internals to any caller.
*   Auth, rate limiting, request ids, structured access logs and metrics, none
    of which existed on an endpoint that can order a production line to stop.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from .config import API_VERSION, SERVICE_VERSION, Settings, get_settings
from .inference import ModelRegistry
from .observability import (METRICS, PredictionAudit, TokenBucket, configure_logging,
                            new_request_id, request_id_var)
from .schemas import (BatchPredictionOut, BatchTelemetryIn, ErrorOut, HealthOut,
                      PredictionOut, ReadinessOut, TelemetryIn)
from .service import ScoringService, UnknownMachineError

log = logging.getLogger("pdm.api")

STATE: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    registry = ModelRegistry(settings)
    registry.load()
    audit = PredictionAudit(settings.prediction_log_path, settings.prediction_log_enabled)
    service = ScoringService(settings, registry, audit)

    STATE["settings"] = settings
    STATE["service"] = service
    STATE["limiter"] = TokenBucket(settings.rate_limit_per_minute)
    METRICS.set_ready(service.ready)

    log.info("service started", extra={
        "version": SERVICE_VERSION, "environment": settings.environment,
        "policy_fingerprint": settings.fingerprint(),
        "onnx": registry.state, "metrics": METRICS.reason,
        "auth": "enabled" if settings.api_key_set else "DISABLED (local only)"})
    if not METRICS.enabled:
        log.warning("prometheus_client missing; /metrics will return 503",
                    extra={"reason": METRICS.reason})
    yield
    log.info("service stopping")


app = FastAPI(
    title="Industrial PdM Service",
    version=SERVICE_VERSION,
    lifespan=lifespan,
    description=(
        "Failure-mode detection for CNC machines.\n\n"
        "**What this is:** an exact detector for the three specified failure modes "
        "(heat dissipation, power envelope, overstrain), plus a closed-form tool-wear "
        "hazard with an explicit horizon.\n\n"
        "**What this is not:** a remaining-useful-life predictor. No learned RUL model "
        "is deployed — the candidate scored R² = −0.104 out of sample. The specification "
        "predicates fire at the moment a limit is breached, so this system provides "
        "*detection*, and *early warning* through envelope margins, but not a lead-time "
        "forecast. See docs/DECISIONS.md."),
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def context_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or new_request_id()
    token = request_id_var.set(rid)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed = time.perf_counter() - started
        METRICS.observe_request(request.url.path, 500, elapsed)
        log.exception("unhandled error", extra={"path": request.url.path})
        request_id_var.reset(token)
        return JSONResponse(
            status_code=500,
            content=ErrorOut(error="Internal error. Quote the request id when reporting.",
                             code="internal_error", request_id=rid).model_dump(),
            headers={"x-request-id": rid})
    elapsed = time.perf_counter() - started
    METRICS.observe_request(request.url.path, response.status_code, elapsed)
    response.headers["x-request-id"] = rid
    log.info("request", extra={"path": request.url.path, "method": request.method,
                               "status": response.status_code,
                               "duration_ms": round(elapsed * 1000, 2)})
    request_id_var.reset(token)
    return response


def _settings() -> Settings:
    return STATE.get("settings") or get_settings()


if get_settings().cors_origins:
    app.add_middleware(
        CORSMiddleware, allow_origins=get_settings().cors_origins,
        allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
async def require_auth(x_api_key: str | None = Header(default=None)) -> str:
    settings = _settings()
    keys = settings.api_key_set
    if not keys:                       # local development only; prod is enforced in config
        return "anonymous"
    if x_api_key is None or x_api_key not in keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="unauthorized")
    return x_api_key[:6]


async def rate_limit(principal: str = Depends(require_auth)) -> str:
    limiter: TokenBucket = STATE["limiter"]          # type: ignore[assignment]
    if not limiter.allow(principal):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="rate limited")
    return principal


def _service() -> ScoringService:
    svc = STATE.get("service")
    if svc is None:
        raise HTTPException(status_code=503, detail="not ready")
    return svc            # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Operational endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthOut, tags=["ops"])
async def health() -> HealthOut:
    """Liveness. Answers only 'is this process alive'."""
    s = _settings()
    return HealthOut(service=s.project_name, version=SERVICE_VERSION,
                     environment=s.environment)


@app.get("/ready", response_model=ReadinessOut, tags=["ops"])
async def ready(response: Response) -> ReadinessOut:
    """Readiness. Answers 'can this process return a correct answer'."""
    svc = STATE.get("service")
    if svc is None:
        response.status_code = 503
        return ReadinessOut(ready=False, detail="starting up", rule_engine=False,
                            hazard_model=False, onnx_classifier="disabled",
                            policy_fingerprint="")
    svc: ScoringService                                   # type: ignore[no-redef]
    info = svc.registry.readiness()
    ok = svc.ready
    if not ok:
        response.status_code = 503
    return ReadinessOut(
        ready=ok, detail=info["detail"], rule_engine=True, hazard_model=True,
        onnx_classifier=info["onnx_classifier"], contract_digest=info["contract_digest"],
        policy_fingerprint=svc.fingerprint)


@app.get("/metrics", tags=["ops"], include_in_schema=False)
async def metrics() -> Response:
    try:
        body, content_type = METRICS.render()
    except RuntimeError as exc:
        return JSONResponse(status_code=503,
                            content={"error": "metrics unavailable", "detail": str(exc)})
    return Response(content=body, media_type=content_type)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@app.post(f"/api/{API_VERSION}/score", response_model=PredictionOut, tags=["scoring"],
          responses={401: {"model": ErrorOut}, 429: {"model": ErrorOut},
                     503: {"model": ErrorOut}})
async def score(reading: TelemetryIn, request: Request,
                principal: str = Depends(rate_limit)) -> PredictionOut:
    svc = _service()
    if not svc.ready:
        raise HTTPException(status_code=503, detail="not ready")
    return svc.score(reading, request_id_var.get())


@app.post(f"/api/{API_VERSION}/score/batch", response_model=BatchPredictionOut,
          tags=["scoring"])
async def score_batch(payload: BatchTelemetryIn, request: Request,
                      principal: str = Depends(rate_limit)) -> BatchPredictionOut:
    """Batch scoring for historian backfills.

    Safe precisely because scoring is stateless: readings in a batch cannot
    influence one another, and batch results are identical to the same readings
    sent one at a time. That property did not hold in the previous service.
    """
    svc = _service()
    if not svc.ready:
        raise HTTPException(status_code=503, detail="not ready")
    rid = request_id_var.get()
    return BatchPredictionOut(
        results=[svc.score(r, f"{rid}-{i}") for i, r in enumerate(payload.readings)])


# ---------------------------------------------------------------------------
# Error handling: stable codes out, detail to the log
# ---------------------------------------------------------------------------
_CODES = {401: "unauthorized", 429: "rate_limited", 503: "not_ready", 422: "validation_error"}


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    rid = request_id_var.get()
    code = _CODES.get(exc.status_code, "internal_error")
    log.warning("request rejected", extra={"status": exc.status_code,
                                           "detail": str(exc.detail),
                                           "path": request.url.path})
    return JSONResponse(status_code=exc.status_code, headers={"x-request-id": rid},
                        content=ErrorOut(error=str(exc.detail), code=code,
                                         request_id=rid).model_dump())


@app.exception_handler(UnknownMachineError)
async def unknown_machine(request: Request, exc: UnknownMachineError):
    """A machine_id outside the asset registry is a caller error, not a fault.

    Rejected rather than scored, because a phantom asset in the audit log
    silently drops those alerts out of the work-order reconciliation.
    """
    rid = request_id_var.get()
    log.warning("unknown machine_id rejected", extra={"machine_id": str(exc)})
    return JSONResponse(
        status_code=422, headers={"x-request-id": rid},
        content=ErrorOut(error="Unknown machine_id: not in the asset registry.",
                         code="unknown_machine", request_id=rid).model_dump())


@app.exception_handler(ValidationError)
async def validation_error(request: Request, exc: ValidationError):
    rid = request_id_var.get()
    log.warning("validation failed", extra={"errors": exc.error_count()})
    return JSONResponse(
        status_code=422, headers={"x-request-id": rid},
        content=ErrorOut(error="Request failed validation. Check sensor ranges and "
                               "product_type (L, M or H).",
                         code="validation_error", request_id=rid).model_dump())
