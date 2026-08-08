"""Model loading and prediction orchestration."""

import logging

import mlflow
import mlflow.pyfunc
import pandas as pd

from api.config import ApiSettings
from api.schemas import PredictRequest, PredictResponse
from features.store import FeatureStore
from ml.features import FEATURE_COLUMNS

logger = logging.getLogger("api.inference")


class ModelUnavailableError(RuntimeError):
    """Raised when the registered model cannot be resolved."""


class FraudDetector:
    """Serves fraud predictions from the MLflow-registered model and Redis features."""

    def __init__(
        self, settings: ApiSettings, store: FeatureStore | None = None
    ) -> None:
        self._settings = settings
        self._store = store or FeatureStore(
            host=settings.redis_host, port=settings.redis_port
        )
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            mlflow.set_tracking_uri(self._settings.mlflow_tracking_uri)
            self._model = mlflow.pyfunc.load_model(
                f"models:/{self._settings.mlflow_model_name}@Production"
            )
            logger.info("Loaded model '%s' from MLflow", self._settings.mlflow_model_name)
        except Exception as exc:  # registry may be unreachable on first boot
            logger.warning("Model not available yet: %s", exc)

    def predict(self, request: PredictRequest) -> PredictResponse:
        if self._model is None:
            raise ModelUnavailableError(
                f"Model '{self._settings.mlflow_model_name}' is not registered yet"
            )
        features = self._store.card_features(
            request.card_id, self._settings.feature_window_seconds
        )
        payload = self._feature_vector(request, features)
        probability = float(self._model.predict(pd.DataFrame([payload]))[0])
        return PredictResponse(
            transaction_id=request.transaction_id,
            is_fraud=probability >= 0.5,
            fraud_probability=probability,
            features=features,
        )

    @staticmethod
    def _feature_vector(request: PredictRequest, features: dict) -> dict:
        # Column order must match FEATURE_COLUMNS used at training time.
        source = {
            "amount": request.amount,
            "channel": request.channel,
            "country": request.country,
            **features,
        }
        return {column: source[column] for column in FEATURE_COLUMNS}
