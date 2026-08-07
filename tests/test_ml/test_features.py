"""Unit tests for ML feature engineering."""

import pytest

pytest.importorskip("pandas")

from ml.features import FEATURE_COLUMNS, build_training_frame  # noqa: E402


def test_feature_columns_are_canonical():
    assert FEATURE_COLUMNS == [
        "amount",
        "channel",
        "country",
        "tx_count",
        "tx_total",
        "tx_avg",
    ]


def test_build_training_frame_shape_and_columns():
    frame = build_training_frame(n_transactions=500, fraud_rate=0.1, seed=7)
    assert list(frame.columns) == FEATURE_COLUMNS + ["is_fraud"]
    assert len(frame) == 500
    assert frame["is_fraud"].isin([0, 1]).all()
    assert frame[FEATURE_COLUMNS].isnull().sum().sum() == 0
    assert frame["tx_count"].min() >= 1


def test_training_frame_has_both_classes():
    frame = build_training_frame(n_transactions=2000, fraud_rate=0.1, seed=3)
    assert set(frame["is_fraud"].unique()) == {0, 1}
