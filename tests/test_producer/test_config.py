"""Unit tests for producer settings."""

import pytest
from pydantic import ValidationError

from producer.config import ProducerSettings


def test_defaults():
    settings = ProducerSettings(bootstrap_servers="localhost:9092")
    assert settings.topic == "transactions.raw"
    assert settings.transactions_per_second == 10.0
    assert settings.fraud_rate == 0.02
    assert settings.max_transactions is None
    assert settings.seed is None


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("REDPANDA_BOOTSTRAP_SERVERS", "broker:29092")
    monkeypatch.setenv("FRAUD_TOPIC", "tx.test")
    monkeypatch.setenv("TRANSACTIONS_PER_SECOND", "25")
    monkeypatch.setenv("MAX_TRANSACTIONS", "100")
    monkeypatch.setenv("SEED", "11")
    settings = ProducerSettings()
    assert settings.bootstrap_servers == "broker:29092"
    assert settings.topic == "tx.test"
    assert settings.transactions_per_second == 25.0
    assert settings.max_transactions == 100
    assert settings.seed == 11


def test_field_names_accepted_programmatically():
    settings = ProducerSettings(
        bootstrap_servers="localhost:9092",
        max_transactions=3,
        transactions_per_second=1000.0,
    )
    assert settings.max_transactions == 3
    assert settings.transactions_per_second == 1000.0


def test_invalid_fraud_rate_rejected():
    with pytest.raises(ValidationError):
        ProducerSettings(bootstrap_servers="localhost:9092", fraud_rate=1.5)
