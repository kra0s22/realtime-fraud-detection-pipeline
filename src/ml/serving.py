"""Shared MLflow model loading for serving (API and alerter)."""

import mlflow
import mlflow.pyfunc


def load_production_model(tracking_uri: str, model_name: str):
    """Load the registered Production model from MLflow.

    Raises on failure so callers decide how to degrade (e.g. serve 503).
    """
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.pyfunc.load_model(f"models:/{model_name}@Production")
