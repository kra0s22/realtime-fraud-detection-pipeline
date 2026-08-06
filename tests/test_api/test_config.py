"""Unit tests for API settings."""

from api.config import ApiSettings


def test_defaults():
    settings = ApiSettings(redis_host="localhost")
    assert settings.redis_port == 6379
    assert settings.mlflow_tracking_uri == "http://mlflow:5000"
    assert settings.mlflow_model_name == "fraud-detector"
    assert settings.feature_window_seconds == 600
    assert settings.port == 8000


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "cache")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracker:5000")
    monkeypatch.setenv("MLFLOW_MODEL_NAME", "fraud-v2")
    settings = ApiSettings()
    assert settings.redis_host == "cache"
    assert settings.redis_port == 6380
    assert settings.mlflow_tracking_uri == "http://tracker:5000"
    assert settings.mlflow_model_name == "fraud-v2"
