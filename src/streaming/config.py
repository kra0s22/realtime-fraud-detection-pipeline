"""Configuration for the PySpark streaming job."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StreamingSettings(BaseSettings):
    """Environment-driven settings for the streaming job."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    bootstrap_servers: str = Field(
        default="redpanda:29092", alias="REDPANDA_BOOTSTRAP_SERVERS"
    )
    topic: str = Field(default="transactions.raw", alias="FRAUD_TOPIC")
    redis_host: str = Field(default="redis", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    delta_table_path: str = Field(
        default="/opt/spark/work-dir/delta/transactions",
        alias="DELTA_TABLE_PATH",
    )
    checkpoint_location: str = Field(
        default="/opt/spark/work-dir/delta/checkpoints/transactions",
        alias="CHECKPOINT_LOCATION",
    )
    feature_ttl_seconds: int = Field(
        default=86400, alias="FEATURE_TTL_SECONDS", ge=60
    )
    processing_interval: str = Field(
        default="5 seconds", alias="PROCESSING_INTERVAL"
    )
