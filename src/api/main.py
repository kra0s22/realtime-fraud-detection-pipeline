"""FastAPI application for low-latency fraud inference."""

import logging

from fastapi import FastAPI, HTTPException
from redis.exceptions import ConnectionError as RedisConnectionError

from api.config import ApiSettings
from api.inference import FraudDetector, ModelUnavailableError
from api.schemas import PredictRequest, PredictResponse
from features.store import FeatureStore

logger = logging.getLogger("api")

settings = ApiSettings()
detector = FraudDetector(settings)
feature_store = FeatureStore(host=settings.redis_host, port=settings.redis_port)

app = FastAPI(title="Fraud Detection API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/alerts")
def alerts(limit: int = 100) -> list:
    try:
        return feature_store.list_alerts(limit)
    except (RedisConnectionError, TimeoutError) as exc:
        logger.warning("Alerts unavailable: %s", exc)
        return []


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        return detector.predict(request)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
