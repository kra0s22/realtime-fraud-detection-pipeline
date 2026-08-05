"""Configuration for the transaction producer."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProducerSettings(BaseSettings):
    """Environment-driven settings for the producer.

    Env vars map to fields by alias (see .env.example); field names are also
    accepted so tests can construct settings programmatically.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    bootstrap_servers: str = Field(
        default="redpanda:29092", alias="REDPANDA_BOOTSTRAP_SERVERS"
    )
    topic: str = Field(default="transactions.raw", alias="FRAUD_TOPIC")
    transactions_per_second: float = Field(
        default=10.0, alias="TRANSACTIONS_PER_SECOND", gt=0
    )
    # When set, the producer stops after this many messages (used in tests/CI).
    max_transactions: int | None = Field(
        default=None, alias="MAX_TRANSACTIONS", ge=1
    )
    fraud_rate: float = Field(default=0.02, alias="FRAUD_RATE", ge=0.0, le=1.0)
    seed: int | None = Field(default=None, alias="SEED")
