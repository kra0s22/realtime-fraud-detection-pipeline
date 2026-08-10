"""Unit and integration tests for batch scoring."""

import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("mlflow")

import mlflow  # noqa: E402

from ml.batch import (  # noqa: E402
    _features_from_raw_transactions,
    _model_feature_columns,
    _resolve_feature_columns,
    score_batch,
)
from ml.config import MlSettings  # noqa: E402
from ml.features import FEATURE_COLUMNS, build_training_frame  # noqa: E402
from ml.train import FraudProbabilityModel, build_pipeline  # noqa: E402
from producer.generator import TransactionGenerator  # noqa: E402


def _register_tiny_model(tmp_path, model_name="batch-test-model") -> None:
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    experiment = mlflow.get_experiment_by_name("batch-tests")
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else mlflow.create_experiment("batch-tests")
    )
    frame = build_training_frame(n_transactions=300, fraud_rate=0.1, seed=3)
    pipeline = build_pipeline()
    pipeline.fit(frame[FEATURE_COLUMNS], frame["is_fraud"])
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
        mlflow.pyfunc.log_model(
            artifact_path="model", python_model=FraudProbabilityModel(pipeline)
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        version = mlflow.register_model(model_uri, model_name)
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(model_name, "Production", version.version)


def _batch_settings(tmp_path, input_path, output_path) -> MlSettings:
    return MlSettings(
        MLFLOW_TRACKING_URI=f"file://{tmp_path}",
        BATCH_MODEL_NAME="batch-test-model",
        BATCH_ALIAS="Production",
        BATCH_INPUT_PATH=str(input_path),
        BATCH_OUTPUT_PATH=str(output_path),
        BATCH_THRESHOLD=0.5,
    )


def test_resolve_feature_columns_splits_and_defaults():
    assert _resolve_feature_columns("a, b") == ["a", "b"]
    assert _resolve_feature_columns(None) is None
    assert _resolve_feature_columns("  ") is None


def test_model_feature_columns_read_from_run_params(tmp_path):
    _register_tiny_model(tmp_path)
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    client = mlflow.tracking.MlflowClient()
    assert _model_feature_columns(client, "batch-test-model", "Production") == FEATURE_COLUMNS


def test_score_batch_writes_probabilities(tmp_path):
    _register_tiny_model(tmp_path)
    frame = build_training_frame(n_transactions=100, fraud_rate=0.1, seed=2)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    frame[FEATURE_COLUMNS].head(20).to_csv(input_path, index=False)
    score_batch(_batch_settings(tmp_path, input_path, output_path))
    out = pd.read_csv(output_path)
    assert len(out) == 20
    assert {"fraud_probability", "is_fraud"} <= set(out.columns)
    assert out["fraud_probability"].between(0, 1).all()
    assert out["is_fraud"].isin([0, 1]).all()


def test_score_batch_missing_columns_raises(tmp_path):
    _register_tiny_model(tmp_path)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame({"amount": [1.0, 2.0]}).to_csv(input_path, index=False)
    with pytest.raises(ValueError, match="missing feature columns"):
        score_batch(_batch_settings(tmp_path, input_path, output_path))


def test_features_from_raw_transactions(tmp_path):
    generator = TransactionGenerator(fraud_rate=0.1, seed=1)
    raw = pd.DataFrame(
        [tx.model_dump() for tx in (generator.generate() for _ in range(50))]
    )
    features = _features_from_raw_transactions(raw)
    assert list(features.columns) == FEATURE_COLUMNS
    assert len(features) == 50
