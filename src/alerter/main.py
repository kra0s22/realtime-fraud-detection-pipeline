"""Realtime fraud alerter: scores streaming transactions and publishes alerts."""

import json
import logging
from datetime import datetime, timezone

import pandas as pd
from confluent_kafka import Consumer, Producer

from alerter.config import AlerterSettings
from features.store import FeatureStore
from ml.features import FEATURE_COLUMNS
from ml.serving import load_production_model

logger = logging.getLogger("alerter")


def build_features(transaction: dict, store: FeatureStore, window_seconds: int) -> dict:
    """Build the FEATURE_COLUMNS vector for a transaction using Redis features."""
    card = store.card_features(transaction["card_id"], window_seconds)
    source = {
        "amount": float(transaction["amount"]),
        "channel": transaction["channel"],
        "country": transaction["country"],
        **card,
    }
    return {column: source[column] for column in FEATURE_COLUMNS}


def score_transaction(
    transaction: dict, model, store: FeatureStore, settings: AlerterSettings
) -> dict | None:
    """Return an alert payload if the fraud probability meets the threshold."""
    features = build_features(transaction, store, settings.feature_window_seconds)
    probability = float(model.predict(pd.DataFrame([features]))[0])
    if probability < settings.threshold:
        return None
    return {
        "transaction_id": transaction["transaction_id"],
        "card_id": transaction["card_id"],
        "amount": features["amount"],
        "channel": transaction["channel"],
        "country": transaction["country"],
        "fraud_probability": probability,
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }


def handle_message(
    value: bytes,
    model,
    store: FeatureStore,
    settings: AlerterSettings,
    producer,
    alerts_topic: str,
) -> dict | None:
    """Parse a Kafka message, score it, and publish an alert if flagged."""
    try:
        transaction = json.loads(value.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("Skipping malformed message")
        return None

    alert = score_transaction(transaction, model, store, settings)
    if alert is None:
        return None

    producer.produce(
        alerts_topic,
        key=alert["transaction_id"],
        value=json.dumps(alert).encode("utf-8"),
        callback=_delivery_callback,
    )
    producer.poll(0)
    store.push_alert(alert)
    return alert


def _delivery_callback(err, msg) -> None:
    if err is not None:
        logger.error("Alert delivery failed for %s: %s", msg.topic(), err)


def main() -> None:
    settings = AlerterSettings()
    model = load_production_model(settings.mlflow_tracking_uri, settings.mlflow_model_name)
    if model is None:
        logger.error("Model not available; alerter cannot start")
        return

    store = FeatureStore(host=settings.redis_host, port=settings.redis_port)
    consumer_conf = {
        "bootstrap.servers": settings.bootstrap_servers,
        "group.id": settings.consumer_group,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
        # Redpanda 24.2 only supports the classic consumer group protocol.
        "group.protocol": "classic",
    }
    producer_conf = {
        "bootstrap.servers": settings.bootstrap_servers,
        "client.id": "fraud-alerter",
    }

    consumer = Consumer(consumer_conf)
    producer = Producer(producer_conf)
    consumer.subscribe([settings.source_topic])
    logger.info(
        "Alerter listening on '%s' -> '%s' (threshold=%s)",
        settings.source_topic,
        settings.alerts_topic,
        settings.threshold,
    )

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error("Consumer error: %s", msg.error())
                continue
            handle_message(
                msg.value(), model, store, settings, producer, settings.alerts_topic
            )
    except KeyboardInterrupt:
        logger.info("Stopping alerter")
    finally:
        consumer.close()
        producer.flush(10)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
