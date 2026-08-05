"""FastAPI application for low-latency fraud inference."""

from fastapi import FastAPI, HTTPException

from api.config import ApiSettings
from api.inference import FraudDetector, ModelUnavailableError
from api.schemas import PredictRequest, PredictResponse

settings = ApiSettings()
detector = FraudDetector(settings)

app = FastAPI(title="Fraud Detection API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        return detector.predict(request)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
