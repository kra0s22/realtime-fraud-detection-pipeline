"""Unit and integration tests for the training pipeline."""

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("mlflow")

import mlflow  # noqa: E402

from ml.config import MlSettings  # noqa: E402
from ml.features import FEATURE_COLUMNS, build_training_frame  # noqa: E402
from ml.train import FraudProbabilityModel, build_pipeline, train_and_register  # noqa: E402


def _local_settings(tmp_path) -> MlSettings:
    # Alias kwargs override the repo .env (by-name kwargs lose to env values).
    return MlSettings(
        MLFLOW_TRACKING_URI=f"file://{tmp_path}",
        MLFLOW_MODEL_NAME="test-fraud-model",
        TRAIN_N_TRANSACTIONS=800,
        FRAUD_RATE=0.1,
        SEED=5,
        TRAIN_TEST_SIZE=0.25,
    )


def test_build_pipeline_fits_and_predicts_probabilities():
    frame = build_training_frame(n_transactions=600, fraud_rate=0.1, seed=1)
    model = build_pipeline()
    model.fit(frame[FEATURE_COLUMNS], frame["is_fraud"])
    probs = FraudProbabilityModel(model).predict(None, frame[FEATURE_COLUMNS].head(10))
    assert probs.shape == (10,)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_train_and_register_logs_to_local_mlflow(tmp_path):
    settings = _local_settings(tmp_path)
    metrics = train_and_register(settings)
    assert "roc_auc" in metrics

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{settings.mlflow_model_name}'")
    assert len(versions) >= 1
    assert "Production" in versions[0].aliases


def test_train_and_register_uses_delta_when_available(tmp_path, monkeypatch):
    settings = _local_settings(tmp_path)
    frame = build_training_frame(n_transactions=400, fraud_rate=0.1, seed=3)
    monkeypatch.setattr("ml.train.frame_from_delta", lambda path: frame)
    metrics = train_and_register(settings)
    assert "roc_auc" in metrics


def test_train_and_register_falls_back_on_bad_delta(tmp_path):
    settings = _local_settings(tmp_path)
    settings.delta_table_path = str(tmp_path / "missing")
    settings.train_use_delta = True
    metrics = train_and_register(settings)
    assert "roc_auc" in metrics
