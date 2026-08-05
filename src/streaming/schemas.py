"""Spark schema for the raw transaction events.

The timestamp is carried as a string during JSON parsing and converted
explicitly in transforms, because the producer emits ISO-8601 UTC timestamps
with a variable-length microsecond fraction.
"""

from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)

TRANSACTION_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("card_id", StringType(), True),
        StructField("merchant_id", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("currency", StringType(), True),
        StructField("country", StringType(), True),
        StructField("channel", StringType(), True),
        StructField("device_id", StringType(), True),
        StructField("is_fraud", BooleanType(), True),
    ]
)

# Matches the producer's serialized timestamp format, e.g. 2026-08-05T13:27:12.070370Z
TIMESTAMP_FORMAT = "yyyy-MM-dd'T'HH:mm:ss.SSSSSS'Z'"
