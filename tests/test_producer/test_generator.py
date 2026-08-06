"""Unit tests for the synthetic transaction generator."""

from datetime import datetime, timezone

from producer.generator import TransactionGenerator


def test_generates_valid_transaction():
    tx = TransactionGenerator(seed=42).generate()
    assert tx.transaction_id
    assert tx.amount > 0
    assert tx.country
    assert tx.channel in ("online", "pos", "atm")


def test_deterministic_with_seed():
    now = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
    gen_a = TransactionGenerator(seed=7)
    gen_b = TransactionGenerator(seed=7)
    for _ in range(10):
        tx_a = gen_a.generate(now)
        tx_b = gen_b.generate(now)
        # transaction_id is intentionally unique per call; everything else must match.
        assert tx_a.model_dump(exclude={"transaction_id"}) == tx_b.model_dump(
            exclude={"transaction_id"}
        )


def test_zero_fraud_rate_never_fraudulent():
    gen = TransactionGenerator(fraud_rate=0.0, seed=1)
    assert all(not gen.generate().is_fraud for _ in range(500))


def test_fraud_rate_within_tolerance():
    gen = TransactionGenerator(fraud_rate=0.1, seed=3)
    sample = [gen.generate().is_fraud for _ in range(5000)]
    observed = sum(sample) / len(sample)
    assert 0.08 <= observed <= 0.12


def test_transaction_ids_are_unique():
    gen = TransactionGenerator(seed=5)
    ids = {gen.generate().transaction_id for _ in range(1000)}
    assert len(ids) == 1000
