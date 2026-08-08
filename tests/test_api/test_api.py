"""API integration tests using FastAPI TestClient."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_503_until_model_registered(client):
    payload = {
        "transaction_id": "tx-1",
        "user_id": "u1",
        "card_id": "c1",
        "merchant_id": "m1",
        "amount": 10.0,
        "currency": "EUR",
        "country": "DE",
        "channel": "pos",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 503


def test_predict_rejects_invalid_payload(client):
    payload = {
        "transaction_id": "tx-1",
        "user_id": "u1",
        "card_id": "c1",
        "merchant_id": "m1",
        "amount": -1,
        "currency": "EUR",
        "country": "DE",
        "channel": "pos",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_alerts_returns_recent_alerts(client, monkeypatch):
    import fakeredis

    import api.main as api_main
    from features.store import FeatureStore

    store = FeatureStore(
        host="localhost",
        port=6379,
        client=fakeredis.FakeRedis(decode_responses=True),
    )
    store.push_alert({"transaction_id": "tx-a", "fraud_probability": 0.9})
    monkeypatch.setattr(api_main, "feature_store", store)

    response = client.get("/alerts")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["transaction_id"] == "tx-a"
