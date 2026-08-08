"""Unit and integration tests for model monitoring."""

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("mlflow")

import mlflow  # noqa: E402

from ml.config import MlSettings  # noqa: E402
from ml.evaluate import evaluate_proba  # noqa: E402
from ml.monitor import evaluate_and_log  # noqa: E402
from ml.train import train_and_register  # noqa: E402


def test_evaluate_proba_perfect_separation():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.9, 0.8])
    metrics = evaluate_proba(y, proba)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_evaluate_proba_below_threshold_counts_as_negative():
    # Positive case scored below the threshold must count as a miss (recall 0).
    y = np.array([0, 1])
    proba = np.array([0.1, 0.3])
    metrics = evaluate_proba(y, proba, threshold=0.5)
    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0


def test_evaluate_and_log_registers_monitoring_run(tmp_path):
    settings = MlSettings(
        mlflow_tracking_uri=f"file://{tmp_path}",
        mlflow_model_name="test-fraud-model",
        n_transactions=800,
        fraud_rate=0.1,
        seed=5,
        test_size=0.25,
    )
    train_and_register(settings)
    metrics = evaluate_and_log(settings)
    assert "roc_auc" in metrics

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name("fraud-detection")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    monitor = runs[runs["tags.mlflow.runName"] == "fraud-detector-monitoring"]
    assert len(monitor) == 1
