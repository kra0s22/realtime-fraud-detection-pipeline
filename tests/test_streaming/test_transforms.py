"""Unit tests for streaming transformations (PySpark local)."""

import pytest

pytest.importorskip("pyspark")


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("test-transforms")
        .getOrCreate()
    )
    yield session
    session.stop()


def _kafka_frame(spark, values):
    return spark.createDataFrame([(v,) for v in values], ["value"])


def test_parse_transactions(spark):
    from streaming.transforms import parse_transactions

    raw = _kafka_frame(
        spark,
        [
            (
                b'{"transaction_id":"tx-1","timestamp":"2026-08-05T13:00:00.000000Z",'
                b'"user_id":"u1","card_id":"c1","merchant_id":"m1","amount":100.5,'
                b'"currency":"EUR","country":"DE","channel":"pos","device_id":"d1",'
                b'"is_fraud":false}'
            )
        ],
    )
    row = parse_transactions(raw).collect()[0]
    assert row.transaction_id == "tx-1"
    assert row.card_id == "c1"
    assert row.amount == 100.5
    assert row.is_fraud is False
    assert row.timestamp is not None


def test_parse_drops_malformed_and_bad_timestamps(spark):
    from streaming.transforms import parse_transactions

    raw = _kafka_frame(
        spark,
        [
            b"not-json",
            b'{"transaction_id":"tx-2","timestamp":"2026-08-05T13:00:00.000000Z",'
            b'"user_id":"u1","card_id":"c1","merchant_id":"m1","amount":20.0,'
            b'"currency":"EUR","country":"DE","channel":"pos"}',
            b'{"transaction_id":"tx-3","timestamp":"not-a-date",'
            b'"user_id":"u1","card_id":"c1","merchant_id":"m1","amount":30.0,'
            b'"currency":"EUR","country":"DE","channel":"pos"}',
        ],
    )
    ids = [r.transaction_id for r in parse_transactions(raw).collect()]
    assert ids == ["tx-2"]
