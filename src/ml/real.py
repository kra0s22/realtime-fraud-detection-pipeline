"""Training and evaluation on the UCI credit card fraud dataset (real-world data)."""

import logging

import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.config import MlSettings
from ml.evaluate import evaluate
from ml.models import build_classifier

logger = logging.getLogger("ml.real")

# Default schema for the UCI Credit Card Fraud dataset.
# V1..V28 are PCA-transformed; Amount is the raw transaction value in euros.
REAL_FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
EXPERIMENT_NAME = "fraud-detection-real"


class RealFraudProbabilityModel(mlflow.pyfunc.PythonModel):
    """Pyfunc wrapper returning fraud probabilities instead of hard labels."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def predict(self, context, model_input):
        return self._pipeline.predict_proba(model_input)[:, 1]


def load_real_dataset(
    path: str,
    feature_columns: list[str] | None = None,
    label_column: str = "Class",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a labeled CSV into (features, labels).

    Defaults to the UCI credit card schema; pass feature_columns and
    label_column to train on any other tabular dataset.
    """
    columns = feature_columns or REAL_FEATURE_COLUMNS
    required = columns + [label_column]
    frame = pd.read_csv(path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return frame[columns], frame[label_column].astype(int)


def _resolve_feature_columns(raw: str | None) -> list[str] | None:
    """Split a comma-separated column list from config; None means the default."""
    if raw is None or not raw.strip():
        return None
    return [column.strip() for column in raw.split(",") if column.strip()]


def build_pipeline(model_type: str = "random_forest") -> Pipeline:
    """Build the classifier for real-world data.

    Random forest is the default: with a ~0.17% fraud rate, logistic
    regression at a 0.5 threshold yields precision ~0.06 / F1 ~0.11 (too many
    false positives), while random forest reaches precision ~0.96 / recall
    ~0.76. Override with REAL_MODEL_TYPE to train any registered model.
    """
    return Pipeline(steps=[("clf", build_classifier(model_type))])


def _ensure_experiment() -> str:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return mlflow.create_experiment(EXPERIMENT_NAME)
    return experiment.experiment_id


def train_and_register_real(settings: MlSettings) -> dict:
    """Train on the real dataset, log metrics to MLflow, and register the model."""
    if not settings.real_data_path:
        raise ValueError("REAL_DATA_PATH is not configured")
    columns = _resolve_feature_columns(settings.real_feature_columns)
    x, y = load_real_dataset(
        settings.real_data_path,
        feature_columns=columns,
        label_column=settings.real_label_column,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=settings.test_size, random_state=settings.seed, stratify=y
    )

    pipeline = build_pipeline(settings.real_model_type)
    pipeline.fit(x_train, y_train)
    metrics = evaluate(pipeline, x_test, y_test)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_id = _ensure_experiment()

    with mlflow.start_run(
        experiment_id=experiment_id, run_name="fraud-detector-training-real"
    ):
        mlflow.log_params(
            {
                "n_transactions": len(x),
                "fraud_rate": float(y.mean()),
                "model": settings.real_model_type,
                "dataset": settings.real_data_path,
                "feature_columns": ",".join(columns or REAL_FEATURE_COLUMNS),
                "label_column": settings.real_label_column,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=RealFraudProbabilityModel(pipeline),
            input_example=x_train.head(1),
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        registered = mlflow.register_model(model_uri, settings.real_model_name)
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            settings.real_model_name, "Production", registered.version
        )

    logger.info(
        "Registered '%s' v%s as Production (real data)",
        settings.real_model_name,
        registered.version,
    )
    return metrics


def main() -> None:
    metrics = train_and_register_real(MlSettings())
    logger.info("Real-data training finished: %s", metrics)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
