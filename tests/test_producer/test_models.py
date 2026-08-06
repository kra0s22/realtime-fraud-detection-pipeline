"""Unit tests for producer domain models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from producer.models import Transaction


def _valid_transaction(**overrides):
    defaults = {
        "user_id": "user-0001",
        "card_id": "card-0001",
        "merchant_id": "merchant-0001",
        "amount": 10.0,
        "currency": "EUR",
        "country": "DE",
        "channel": "pos",
    }
    defaults.update(overrides)
    return Transaction(**defaults)


def test_valid_transaction_rounds_amount():
    tx = _valid_transaction(amount=123.456)
    assert tx.amount == 123.46


def test_defaults_are_applied():
    tx = _valid_transaction()
    assert tx.transaction_id
    assert isinstance(tx.timestamp, datetime)
    assert tx.timestamp.tzinfo == timezone.utc
    assert tx.is_fraud is False
    assert tx.device_id is None


def test_string_enums_are_coerced():
    tx = _valid_transaction(currency="USD", channel="atm")
    assert tx.currency == "USD"
    assert tx.channel == "atm"


def test_amount_must_be_positive():
    with pytest.raises(ValidationError):
        _valid_transaction(amount=0)


def test_negative_amount_rejected():
    with pytest.raises(ValidationError):
        _valid_transaction(amount=-5.0)
