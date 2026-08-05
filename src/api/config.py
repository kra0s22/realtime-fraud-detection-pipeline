"""Configuration for the FastAPI inference service."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """Environment-driven settings for the inference service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
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
    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8000, alias="API_PORT")
