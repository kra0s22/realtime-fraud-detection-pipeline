"""Evaluate the Production model on fresh data and log metrics to MLflow."""

import logging

import mlflow
import numpy as np
import pandas as pd

from ml.config import MlSettings
from ml.evaluate import evaluate_proba
from ml.features import FEATURE_COLUMNS, build_training_frame
from ml.serving import load_production_model
from ml.train import ensure_experiment

logger = logging.getLogger("ml.monitor")

_MONITOR_RUN_NAME = "fraud-detector-monitoring"


def evaluate_and_log(settings: MlSettings) -> dict:
    """Score the Production model on fresh data and log metrics to MLflow."""
    model = load_production_model(settings.mlflow_tracking_uri, settings.mlflow_model_name)
    if model is None:
        raise RuntimeError("Production model not available for evaluation")

    # A different seed than training simulates unseen events (held-out drift check).
    frame = build_training_frame(
        n_transactions=settings.n_transactions,
        fraud_rate=settings.fraud_rate,
        seed=settings.seed + 1000,
    )
    x = frame[FEATURE_COLUMNS]
    y = frame["is_fraud"].astype(int)
    proba = np.asarray(model.predict(pd.DataFrame(x)))
    metrics = evaluate_proba(y, proba)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    with mlflow.start_run(experiment_id=ensure_experiment(), run_name=_MONITOR_RUN_NAME):
        mlflow.log_params(
            {
                "model": settings.mlflow_model_name,
                "n_transactions": settings.n_transactions,
                "seed": settings.seed + 1000,
                "threshold": 0.5,
            }
        )
        mlflow.log_metrics(metrics)

    logger.info("Monitoring metrics: %s", metrics)
    return metrics


def main() -> None:
    settings = MlSettings()
    metrics = evaluate_and_log(settings)
    logger.info("Monitoring finished: %s", metrics)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
