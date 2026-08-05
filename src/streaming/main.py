"""PySpark Structured Streaming job: Redpanda -> Delta Lake + Redis feature store."""

import logging

from pyspark.sql import SparkSession

from features.store import FeatureStore
from streaming.config import StreamingSettings
from streaming.transforms import parse_transactions

logger = logging.getLogger("streaming")


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("fraud-streaming")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )


def write_features_to_redis(batch_df, _batch_id: int, store: FeatureStore) -> None:
    """Index each transaction in the Redis feature store (driver-side)."""
    for row in batch_df.collect():
        store.record_transaction(
            card_id=row.card_id,
            tx_id=row.transaction_id,
            amount=float(row.amount),
            ts_epoch=row.timestamp.timestamp(),
        )


def main() -> None:
    settings = StreamingSettings()
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    store = FeatureStore(
        host=settings.redis_host,
        port=settings.redis_port,
        ttl_seconds=settings.feature_ttl_seconds,
    )

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.bootstrap_servers)
        .option("subscribe", settings.topic)
        .option("startingOffsets", "latest")
        .load()
    )

    transactions = parse_transactions(raw)

    transactions.writeStream.format("delta").outputMode("append").option(
        "checkpointLocation", settings.checkpoint_location
    ).trigger(processingTime=settings.processing_interval).start(
        settings.delta_table_path
    )

    transactions.writeStream.foreachBatch(
        lambda df, bid: write_features_to_redis(df, bid, store)
    ).outputMode("update").option(
        "checkpointLocation", settings.checkpoint_location + "-redis"
    ).trigger(processingTime=settings.processing_interval).start()

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
