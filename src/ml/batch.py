"""Batch scoring: run any registered model over a CSV of features."""

import logging

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

from ml.config import MlSettings
from ml.features import FEATURE_COLUMNS

logger = logging.getLogger("ml.batch")

_REQUIRED_RAW_COLUMNS = {"card_id", "transaction_id", "amount", "channel", "country"}


def _resolve_feature_columns(raw: str | None) -> list[str] | None:
    """Split a comma-separated column list from config; None means unset."""
    if raw is None or not raw.strip():
        return None
    return [column.strip() for column in raw.split(",") if column.strip()]


def _model_feature_columns(
    client: mlflow.tracking.MlflowClient, model_name: str, alias: str
) -> list[str] | None:
    """Read the feature contract logged at training time for a registered model."""
    registered = client.get_registered_model(model_name)
    version = registered.aliases.get(alias)
    if version is None:
        return None
    run_id = client.get_model_version(model_name, version).run_id
    return _resolve_feature_columns(client.get_run(run_id).data.params.get("feature_columns"))


def _features_from_raw_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute the base model's velocity features from raw transactions."""
    from ml.features import _with_card_features

    missing = _REQUIRED_RAW_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Raw transactions missing columns for feature building: {sorted(missing)}"
        )
    if "is_fraud" not in frame.columns:
        frame["is_fraud"] = 0
    return _with_card_features(frame)[FEATURE_COLUMNS]


def score_batch(settings: MlSettings) -> str:
    """Score a CSV with a registered model and write fraud probabilities.

    The model is expected to return fraud probabilities (as all models in this
    repo do via their pyfunc wrapper). The expected input columns are read from
    the model's logged ``feature_columns`` contract, so the same command works
    for the live model, the real-data model, or any registered model.
    """
    if not settings.batch_input_path or not settings.batch_output_path:
        raise ValueError("BATCH_INPUT_PATH and BATCH_OUTPUT_PATH are required")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    model_name = settings.batch_model_name or settings.mlflow_model_name

    feature_columns = _resolve_feature_columns(
        settings.batch_feature_columns
    ) or _model_feature_columns(client, model_name, settings.batch_alias)

    model = mlflow.pyfunc.load_model(f"models:/{model_name}@{settings.batch_alias}")

    frame = pd.read_csv(settings.batch_input_path)
    if settings.batch_build_features:
        features = _features_from_raw_transactions(frame)
    else:
        if feature_columns is None:
            raise ValueError(
                "Could not read the model's feature_columns. Set BATCH_FEATURE_COLUMNS."
            )
        missing = [column for column in feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(
                f"Input CSV missing feature columns {missing}. Expected: {feature_columns}"
            )
        features = frame[feature_columns]

    probabilities = np.asarray(model.predict(features), dtype=float)

    output = frame.copy()
    output["fraud_probability"] = probabilities
    output["is_fraud"] = (probabilities >= settings.batch_threshold).astype(int)
    output.to_csv(settings.batch_output_path, index=False)
    logger.info(
        "Scored %d rows with '%s@%s' -> %s",
        len(features),
        model_name,
        settings.batch_alias,
        settings.batch_output_path,
    )
    return str(settings.batch_output_path)


def main() -> None:
    score_batch(MlSettings())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
