"""Training, evaluation, and MLflow registration of the fraud detection model."""

import logging

import mlflow
import mlflow.pyfunc
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.config import MlSettings
from ml.evaluate import evaluate
from ml.features import FEATURE_COLUMNS, build_training_frame

logger = logging.getLogger("ml.train")

_NUMERIC = ["amount", "tx_count", "tx_total", "tx_avg"]
_CATEGORICAL = ["channel", "country"]
EXPERIMENT_NAME = "fraud-detection"


class FraudProbabilityModel(mlflow.pyfunc.PythonModel):
    """Pyfunc wrapper returning fraud probabilities instead of hard labels."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def predict(self, context, model_input):
        return self._pipeline.predict_proba(model_input)[:, 1]


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), _NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), _CATEGORICAL),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            # class_weight balances the heavily skewed fraud rate.
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def ensure_experiment() -> str:
    """Return the fraud-detection experiment id, creating it if needed."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return mlflow.create_experiment(EXPERIMENT_NAME)
    return experiment.experiment_id


def train_and_register(settings: MlSettings) -> dict:
    """Train the model, log it to MLflow, and promote it to Production."""
    frame = build_training_frame(
        n_transactions=settings.n_transactions,
        fraud_rate=settings.fraud_rate,
        seed=settings.seed,
    )
    x = frame[FEATURE_COLUMNS]
    y = frame["is_fraud"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=settings.test_size, random_state=settings.seed, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    metrics = evaluate(pipeline, x_test, y_test)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_id = ensure_experiment()

    with mlflow.start_run(experiment_id=experiment_id, run_name="fraud-detector-training"):
        mlflow.log_params(
            {
                "n_transactions": settings.n_transactions,
                "fraud_rate": settings.fraud_rate,
                "model": "logistic_regression",
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=FraudProbabilityModel(pipeline),
            input_example=x_train.head(1),
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        registered = mlflow.register_model(model_uri, settings.mlflow_model_name)
        client = mlflow.tracking.MlflowClient()
        # Aliases replace the deprecated registry stages (removed in a future MLflow release).
        client.set_registered_model_alias(
            settings.mlflow_model_name, "Production", registered.version
        )

    logger.info(
        "Registered '%s' v%s as Production", settings.mlflow_model_name, registered.version
    )
    return metrics


def main() -> None:
    metrics = train_and_register(MlSettings())
    logger.info("Training finished: %s", metrics)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
