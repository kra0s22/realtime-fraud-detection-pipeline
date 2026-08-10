"""Unit tests for the real-world (UCI credit card) training path."""

import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("mlflow")

from ml.config import MlSettings  # noqa: E402
from ml.real import (  # noqa: E402
    REAL_FEATURE_COLUMNS,
    _resolve_feature_columns,
    build_pipeline,
    load_real_dataset,
    train_and_register_real,
)


def _write_sample_csv(path, n=50, with_class=True) -> None:
    rows = []
    for i in range(n):
        row = {f"V{j}": float(j + i) for j in range(1, 29)}
        row["Amount"] = 10.0 + i
        if with_class:
            # ~5% fraud so the balanced classifier sees both classes.
            row["Class"] = 1 if i % 20 == 0 else 0
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_real_dataset_returns_features_and_labels(tmp_path):
    path = tmp_path / "creditcard.csv"
    _write_sample_csv(path)
    x, y = load_real_dataset(str(path))
    assert list(x.columns) == REAL_FEATURE_COLUMNS
    assert len(x) == 50
    assert set(y.unique()) <= {0, 1}


def test_load_real_dataset_raises_on_missing_class(tmp_path):
    path = tmp_path / "creditcard.csv"
    _write_sample_csv(path, with_class=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_real_dataset(str(path))


def test_load_real_dataset_with_custom_schema(tmp_path):
    path = tmp_path / "custom.csv"
    pd.DataFrame(
        {"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0], "fraud": [0, 1, 0]}
    ).to_csv(path, index=False)
    x, y = load_real_dataset(
        str(path), feature_columns=["f1", "f2"], label_column="fraud"
    )
    assert list(x.columns) == ["f1", "f2"]
    assert y.tolist() == [0, 1, 0]


def test_resolve_feature_columns_splits_and_defaults():
    assert _resolve_feature_columns("a, b, c") == ["a", "b", "c"]
    assert _resolve_feature_columns(None) is None
    assert _resolve_feature_columns("  ") is None


def test_build_pipeline_fits_on_real_features(tmp_path):
    path = tmp_path / "creditcard.csv"
    _write_sample_csv(path, n=80)
    x, y = load_real_dataset(str(path))
    model = build_pipeline()
    model.fit(x, y)
    probs = model.predict_proba(x.head(5))[:, 1]
    assert ((probs >= 0) & (probs <= 1)).all()


def test_train_and_register_real_logs_to_local_mlflow(tmp_path):
    path = tmp_path / "creditcard.csv"
    _write_sample_csv(path, n=120)
    # Alias kwargs override the repo .env (by-name kwargs lose to env values).
    settings = MlSettings(
        MLFLOW_TRACKING_URI=f"file://{tmp_path}",
        REAL_DATA_PATH=str(path),
        REAL_MODEL_NAME="test-fraud-real",
        SEED=7,
        TRAIN_TEST_SIZE=0.25,
    )
    metrics = train_and_register_real(settings)
    assert "roc_auc" in metrics
