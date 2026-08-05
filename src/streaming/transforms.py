"""Streaming transformations: Kafka JSON -> structured transactions."""

from pyspark.sql import DataFrame, functions as F

from streaming.schemas import TIMESTAMP_FORMAT, TRANSACTION_SCHEMA


def parse_transactions(raw: DataFrame) -> DataFrame:
    """Parse the Kafka 'value' column into a structured transaction frame.

    Malformed records (null transaction_id) and unparseable timestamps are
    dropped rather than failing the batch.
    """
    parsed = raw.selectExpr("CAST(value AS STRING) AS json")
    parsed = parsed.select(
        F.from_json(F.col("json"), TRANSACTION_SCHEMA).alias("tx")
    ).select("tx.*")
    parsed = parsed.filter(F.col("transaction_id").isNotNull())
    return parsed.withColumn(
        "timestamp", F.to_timestamp(F.col("timestamp"), TIMESTAMP_FORMAT)
    ).filter(F.col("timestamp").isNotNull())
