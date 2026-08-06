"""Unit tests for the fraud detector inference orchestration."""

import time

import pytest

pytest.importorskip("fakeredis")
pytest.importorskip("mlflow")

import fakeredis  # noqa: E402

from api.config import ApiSettings  # noqa: E402
from api.inference import FraudDetector, ModelUnavailableError  # noqa: E402
from api.schemas import PredictRequest  # noqa: E402
from features.store import FeatureStore  # noqa: E402


def _detector():
    import fakeredis as _fakeredis

    store = FeatureStore(
        host="localhost",
        port=6379,
        client=_fakeredis.FakeRedis(decode_responses=True),
    )
    return FraudDetector(ApiSettings(redis_host="localhost"), store=store), store


def _request(**overrides):
    defaults = {
        "transaction_id": "tx-1",
        "user_id": "u1",
        "card_id": "c1",
        "merchant_id": "m1",
        "amount": 10.0,
        "currency": "EUR",
        "country": "DE",
        "channel": "pos",
    }
    defaults.update(overrides)
    return PredictRequest(**defaults)


class FakeModel:
    def predict(self, df):
        return [0.9]


def test_predict_uses_model_and_features():
    detector, store = _detector()
    store.record_transaction("c1", "tx-0", 50.0, ts_epoch=time.time())
    detector._model = FakeModel()

    response = detector.predict(_request())
    assert response.is_fraud is True
    assert response.fraud_probability == 0.9
    assert response.features["tx_count"] == 1
    assert response.features["tx_total"] == 50.0


def test_predict_raises_when_model_unavailable():
    detector, _ = _detector()
    assert detector._model is None
    with pytest.raises(ModelUnavailableError):
        detector.predict(_request())
