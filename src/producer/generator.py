"""Synthetic transaction generator with configurable fraud injection."""

import random
from datetime import datetime, timezone

from producer.models import Channel, Currency, Transaction

_USERS = [f"user-{i:04d}" for i in range(1, 101)]
_CARDS = [f"card-{i:04d}" for i in range(1, 101)]
_MERCHANTS = [f"merchant-{i:04d}" for i in range(1, 51)]
_DEVICES = [f"device-{i:04d}" for i in range(1, 51)]
_COUNTRIES = ["DE", "FR", "ES", "IT", "NL", "BE", "AT", "PT"]


class TransactionGenerator:
    """Produces realistic transactions; a configurable fraction are fraudulent.

    Fraudulent transactions use higher amounts and more frequently target
    online channels and foreign countries, giving the downstream model signal.
    """

    def __init__(self, fraud_rate: float = 0.02, seed: int | None = None) -> None:
        self._fraud_rate = fraud_rate
        self._rng = random.Random(seed)

    def generate(self, now: datetime | None = None) -> Transaction:
        is_fraud = self._rng.random() < self._fraud_rate
        timestamp = now or datetime.now(timezone.utc)

        if is_fraud:
            amount = self._rng.uniform(500.0, 5000.0)
            channel = Channel.ONLINE if self._rng.random() < 0.8 else Channel.POS
            country = self._rng.choice(_COUNTRIES)
        else:
            amount = self._rng.uniform(1.0, 500.0)
            channel = self._rng.choices(
                [Channel.POS, Channel.ONLINE, Channel.ATM], weights=[6, 3, 1]
            )[0]
            country = "DE"

        return Transaction(
            timestamp=timestamp,
            user_id=self._rng.choice(_USERS),
            card_id=self._rng.choice(_CARDS),
            merchant_id=self._rng.choice(_MERCHANTS),
            amount=amount,
            currency=self._rng.choice(
                [Currency.EUR, Currency.EUR, Currency.EUR, Currency.USD, Currency.GBP]
            ),
            country=country,
            channel=channel,
            device_id=self._rng.choice(_DEVICES),
            is_fraud=is_fraud,
        )
