"""Unit tests for the Redis feature store using fakeredis."""

import pytest

from features.store import FeatureStore

pytest.importorskip("fakeredis")


@pytest.fixture()
def store():
    import fakeredis

    client = fakeredis.FakeRedis(decode_responses=True)
    return FeatureStore(host="localhost", port=6379, client=client)


def test_record_and_aggregate(store):
    store.record_transaction("card-0001", "tx-1", 10.0, ts_epoch=1000)
    store.record_transaction("card-0001", "tx-2", 20.0, ts_epoch=1100)
    assert store.card_features("card-0001", window_seconds=3600, now=2000) == {
        "tx_count": 2,
        "tx_total": 30.0,
        "tx_avg": 15.0,
    }


def test_window_excludes_old_transactions(store):
    store.record_transaction("card-0001", "tx-1", 10.0, ts_epoch=0)
    store.record_transaction("card-0001", "tx-2", 20.0, ts_epoch=500)
    # Window [400, 1000] includes only tx-2.
    features = store.card_features("card-0001", window_seconds=600, now=1000)
    assert features == {"tx_count": 1, "tx_total": 20.0, "tx_avg": 20.0}


def test_unknown_card_returns_zeros(store):
    assert store.card_features("card-9999", window_seconds=600, now=1000) == {
        "tx_count": 0,
        "tx_total": 0.0,
        "tx_avg": 0.0,
    }


def test_features_isolated_per_card(store):
    store.record_transaction("card-0001", "tx-1", 10.0, ts_epoch=1500)
    store.record_transaction("card-0002", "tx-2", 50.0, ts_epoch=1500)
    assert store.card_features("card-0002", window_seconds=600, now=2000)[
        "tx_total"
    ] == 50.0
