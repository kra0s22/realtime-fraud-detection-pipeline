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
