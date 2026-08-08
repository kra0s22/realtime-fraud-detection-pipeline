"""Configuration for the ML training pipeline."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MlSettings(BaseSettings):
    """Environment-driven settings for model training."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    mlflow_tracking_uri: str = Field(
        default="http://mlflow:5000", alias="MLFLOW_TRACKING_URI"
    )
    mlflow_model_name: str = Field(
        default="fraud-detector", alias="MLFLOW_MODEL_NAME"
    )
    n_transactions: int = Field(
        default=20000, alias="TRAIN_N_TRANSACTIONS", ge=100
    )
    fraud_rate: float = Field(default=0.02, alias="FRAUD_RATE", ge=0.0, le=1.0)
    seed: int = Field(default=42, alias="SEED")
    test_size: float = Field(
        default=0.2, alias="TRAIN_TEST_SIZE", gt=0.0, lt=1.0
    )
    delta_table_path: str | None = Field(
        default="/opt/spark/work-dir/delta/transactions",
        alias="DELTA_TABLE_PATH",
    )
    train_use_delta: bool = Field(default=True, alias="TRAIN_USE_DELTA")
