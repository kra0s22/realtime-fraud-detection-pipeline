"""Domain models for the transaction pipeline."""

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, field_serializer, field_validator


class Channel(str, Enum):
    ONLINE = "online"
    POS = "pos"
    ATM = "atm"


class Currency(str, Enum):
    EUR = "EUR"
    USD = "USD"
    GBP = "GBP"


class Transaction(BaseModel):
    """A single payment transaction emitted to the raw topic."""

    transaction_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: str
    card_id: str
    merchant_id: str
    amount: float = Field(gt=0)
    currency: Currency
    country: str
    channel: Channel
    device_id: str | None = None
    is_fraud: bool = False

    @field_validator("amount")
    @classmethod
    def _round_amount(cls, value: float) -> float:
        return round(value, 2)

    @field_serializer("timestamp")
    def _serialize_timestamp(self, value: datetime) -> str:
        # Fixed 6-digit microsecond + 'Z' so Spark can parse it deterministically.
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
