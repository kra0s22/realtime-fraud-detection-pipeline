"""Unit tests for ML feature engineering."""

import pandas as pd
import pytest

pytest.importorskip("pandas")

from ml.features import (  # noqa: E402
    FEATURE_COLUMNS,
    _with_card_features,
    build_training_frame,
    frame_from_delta,
)


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


def test_with_card_features_computes_velocity():
    raw = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3"],
            "card_id": ["c1", "c1", "c2"],
            "amount": [10.0, 20.0, 50.0],
            "channel": ["pos", "online", "atm"],
            "country": ["DE", "NL", "DE"],
            "is_fraud": [0, 1, 0],
        }
    )
    frame = _with_card_features(raw)
    assert list(frame.columns) == FEATURE_COLUMNS + ["is_fraud"]
    t1 = frame[frame["amount"] == 10.0].iloc[0]
    assert t1["tx_count"] == 2
    assert t1["tx_total"] == 30.0
    assert t1["tx_avg"] == 15.0


def test_frame_from_delta_roundtrip(tmp_path):
    from deltalake import write_deltalake

    raw = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3"],
            "card_id": ["c1", "c1", "c2"],
            "amount": [10.0, 20.0, 50.0],
            "channel": ["pos", "online", "atm"],
            "country": ["DE", "NL", "DE"],
            "is_fraud": [0, 1, 0],
        }
    )
    path = str(tmp_path / "tx")
    write_deltalake(path, raw)
    frame = frame_from_delta(path)
    assert list(frame.columns) == FEATURE_COLUMNS + ["is_fraud"]
    assert len(frame) == 3
    assert frame["is_fraud"].sum() == 1


def test_frame_from_delta_missing_path_raises(tmp_path):
    from deltalake.exceptions import TableNotFoundError

    with pytest.raises(TableNotFoundError):
        frame_from_delta(str(tmp_path / "missing"))
