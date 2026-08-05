"""Request/response models for the inference API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    transaction_id: str
    user_id: str
    card_id: str
    merchant_id: str
    amount: float = Field(gt=0)
    currency: str
    country: str
    channel: str
    device_id: str | None = None


class PredictResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    features: dict[str, float | int]
