"""Configuration for the fraud alerter service."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AlerterSettings(BaseSettings):
    """Environment-driven settings for the fraud alerter."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    bootstrap_servers: str = Field(
        default="redpanda:29092", alias="REDPANDA_BOOTSTRAP_SERVERS"
    )
    source_topic: str = Field(default="transactions.raw", alias="FRAUD_TOPIC")
    alerts_topic: str = Field(
        default="transactions.alerts", alias="ALERTS_TOPIC"
    )
    consumer_group: str = Field(
        default="fraud-alerter", alias="ALERTER_GROUP"
    )
    redis_host: str = Field(default="redis", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    mlflow_tracking_uri: str = Field(
        default="http://mlflow:5000", alias="MLFLOW_TRACKING_URI"
    )
    mlflow_model_name: str = Field(
        default="fraud-detector", alias="MLFLOW_MODEL_NAME"
    )
    feature_window_seconds: int = Field(
        default=600, alias="FEATURE_WINDOW_SECONDS", ge=1
    )
    threshold: float = Field(default=0.5, alias="ALERT_THRESHOLD", ge=0.0, le=1.0)
