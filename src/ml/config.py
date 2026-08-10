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
    real_data_path: str | None = Field(
        default="/app/data/creditcard.csv", alias="REAL_DATA_PATH"
    )
    real_model_name: str = Field(
        default="fraud-detector-real", alias="REAL_MODEL_NAME"
    )
    # Live-demo (synthetic) model type; any CLASSIFIERS entry is valid.
    model_type: str = Field(
        default="logistic_regression", alias="MODEL_TYPE"
    )
    # Real-data model type; random forest handles the ~0.17% fraud rate best.
    real_model_type: str = Field(
        default="random_forest", alias="REAL_MODEL_TYPE"
    )
    # Comma-separated feature columns for the real dataset; None -> UCI default.
    real_feature_columns: str | None = Field(
        default=None, alias="REAL_FEATURE_COLUMNS"
    )
    real_label_column: str = Field(default="Class", alias="REAL_LABEL_COLUMN")
    # Batch scoring (src/ml/batch.py)
    batch_input_path: str | None = Field(default=None, alias="BATCH_INPUT_PATH")
    batch_output_path: str | None = Field(default=None, alias="BATCH_OUTPUT_PATH")
    batch_model_name: str | None = Field(default=None, alias="BATCH_MODEL_NAME")
    batch_alias: str = Field(default="Production", alias="BATCH_ALIAS")
    batch_threshold: float = Field(
        default=0.5, alias="BATCH_THRESHOLD", ge=0.0, le=1.0
    )
    batch_feature_columns: str | None = Field(
        default=None, alias="BATCH_FEATURE_COLUMNS"
    )
    batch_build_features: bool = Field(
        default=False, alias="BATCH_BUILD_FEATURES"
    )
