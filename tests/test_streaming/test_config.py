"""Unit tests for streaming settings."""

from streaming.config import StreamingSettings


def test_defaults():
    settings = StreamingSettings(bootstrap_servers="localhost:9092")
    assert settings.topic == "transactions.raw"
    assert settings.redis_host == "redis"
    assert settings.redis_port == 6379
    assert settings.delta_table_path == "/opt/spark/work-dir/delta/transactions"
    assert settings.processing_interval == "5 seconds"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("REDPANDA_BOOTSTRAP_SERVERS", "broker:29092")
    monkeypatch.setenv("FRAUD_TOPIC", "tx.stream")
    monkeypatch.setenv("DELTA_TABLE_PATH", "/tmp/delta/tx")
    monkeypatch.setenv("CHECKPOINT_LOCATION", "/tmp/delta/cp")
    monkeypatch.setenv("FEATURE_TTL_SECONDS", "7200")
    settings = StreamingSettings()
    assert settings.bootstrap_servers == "broker:29092"
    assert settings.topic == "tx.stream"
    assert settings.delta_table_path == "/tmp/delta/tx"
    assert settings.checkpoint_location == "/tmp/delta/cp"
    assert settings.feature_ttl_seconds == 7200
