"""Unit tests for alerter settings."""

from alerter.config import AlerterSettings


def test_defaults():
    settings = AlerterSettings(bootstrap_servers="localhost:9092")
    assert settings.source_topic == "transactions.raw"
    assert settings.alerts_topic == "transactions.alerts"
    assert settings.consumer_group == "fraud-alerter"
    assert settings.threshold == 0.5


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("ALERTS_TOPIC", "tx.alerts")
    monkeypatch.setenv("ALERT_THRESHOLD", "0.7")
    monkeypatch.setenv("ALERTER_GROUP", "my-group")
    settings = AlerterSettings(bootstrap_servers="localhost:9092")
    assert settings.alerts_topic == "tx.alerts"
    assert settings.threshold == 0.7
    assert settings.consumer_group == "my-group"
