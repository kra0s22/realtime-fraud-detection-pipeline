"""Unit tests for the fraud alerter scoring and message handling."""

import json

import pytest

pytest.importorskip("fakeredis")

from alerter.config import AlerterSettings  # noqa: E402
from alerter.main import build_features, handle_message, score_transaction  # noqa: E402
from features.store import FeatureStore  # noqa: E402


class FakeModel:
    def __init__(self, probability):
        self._probability = probability

    def predict(self, df):
        return [self._probability]


class FakeProducer:
    def __init__(self):
        self.produced = []

    def produce(self, topic, key, value, callback=None):
        self.produced.append((topic, key, value))

    def poll(self, timeout=0):
        return 0


def _store():
    import fakeredis

    return FeatureStore(
        host="localhost",
        port=6379,
        client=fakeredis.FakeRedis(decode_responses=True),
    )


def _settings():
    return AlerterSettings(
        bootstrap_servers="localhost:9092",
        redis_host="localhost",
        mlflow_tracking_uri="http://localhost:5000",
        mlflow_model_name="test-model",
    )


def _tx():
    return {
        "transaction_id": "tx-1",
        "user_id": "u1",
        "card_id": "c1",
        "merchant_id": "m1",
        "amount": 4500.0,
        "currency": "EUR",
        "country": "NL",
        "channel": "online",
        "is_fraud": True,
    }


def test_score_transaction_returns_alert_above_threshold():
    alert = score_transaction(_tx(), FakeModel(0.9), _store(), _settings())
    assert alert is not None
    assert alert["transaction_id"] == "tx-1"
    assert alert["card_id"] == "c1"
    assert alert["fraud_probability"] == 0.9


def test_score_transaction_returns_none_below_threshold():
    assert score_transaction(_tx(), FakeModel(0.1), _store(), _settings()) is None


def test_handle_message_publishes_alert_and_records_in_redis():
    settings = _settings()
    store = _store()
    producer = FakeProducer()
    alert = handle_message(
        json.dumps(_tx()).encode("utf-8"),
        FakeModel(0.95),
        store,
        settings,
        producer,
        "transactions.alerts",
    )
    assert alert is not None
    assert len(producer.produced) == 1
    topic, key, value = producer.produced[0]
    assert topic == "transactions.alerts"
    assert key == "tx-1"
    assert json.loads(value)["fraud_probability"] == 0.95
    stored = store.list_alerts()
    assert len(stored) == 1
    assert stored[0]["transaction_id"] == "tx-1"


def test_handle_message_skips_malformed():
    store = _store()
    producer = FakeProducer()
    assert handle_message(b"not-json", FakeModel(0.9), store, _settings(), producer, "t") is None
    assert len(producer.produced) == 0


def test_build_features_uses_feature_columns():
    features = build_features(_tx(), _store(), window_seconds=600)
    assert list(features) == [
        "amount",
        "channel",
        "country",
        "tx_count",
        "tx_total",
        "tx_avg",
    ]
    assert features["amount"] == 4500.0
    assert features["channel"] == "online"
