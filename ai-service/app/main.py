"""FastAPI entry point; model loading and routes belong here."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.auth import verify_api_key
from app.config import get_settings
from app.inference import load_model, predict_outcome
from app.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load the ML model at startup; log but tolerate failures."""
    settings = get_settings()

    logging.basicConfig(
        level=settings.LOG_LEVEL.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    application.state.model = None
    application.state.model_version = None

    try:
        application.state.model = load_model(settings.MODEL_PATH)
        application.state.model_version = (
            settings.MODEL_VERSION or Path(settings.MODEL_PATH).stem
        )
        logger.info(
            "Model loaded: version=%s", application.state.model_version
        )
    except Exception:
        logger.exception("Failed to load model — service starts in degraded mode")

    if not settings.AI_API_KEY:
        logger.warning("/predict is UNPROTECTED — AI_API_KEY is empty")

    yield


app = FastAPI(title="Renal Insight AI Service", lifespan=lifespan)


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return 500 for unexpected errors without leaking details."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error("Unhandled %s for request_id=%s", type(exc).__name__, request_id)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def _get_model(request: Request) -> object:
    """Dependency that returns the loaded model or raises 503."""
    if request.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return request.state.model


@app.middleware("http")
async def _inject_model_and_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Copy app-level model state into request.state for each request."""
    request.state.model = app.state.model
    request.state.model_version = app.state.model_version
    request.state.request_id = uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    request: Request,
    _auth: None = Depends(verify_api_key),
    model: object = Depends(_get_model),
) -> PredictionResponse:
    """Run CKD risk prediction for a single patient row."""
    outcome = predict_outcome(payload, model)
    logger.info("request_id=%s outcome=%s", request.state.request_id, outcome)
    return PredictionResponse(outcome=outcome)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    """Liveness / readiness probe."""
    if request.state.model is None:
        return JSONResponse(status_code=503, content={"detail": "Model not loaded"})  # type: ignore[return-value]
    return {"status": "UP", "modelVersion": request.state.model_version}
