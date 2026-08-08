"""Redis-backed online feature store shared by the streaming job and the API."""

import json
import time

import redis

_ALERTS_KEY = "fraud:alerts"


class FeatureStore:
    """Per-card transaction history stored as a time-indexed set.

    Key layout:
      fraud:card:{card_id}:txs  -> ZSET   (member=transaction_id, score=unix timestamp)
      fraud:tx:{transaction_id} -> STRING (amount)
    """

    def __init__(
        self,
        host: str,
        port: int,
        db: int = 0,
        ttl_seconds: int | None = None,
        client: redis.Redis | None = None,
    ) -> None:
        self._client = client or redis.Redis(
            host=host, port=port, db=db, decode_responses=True
        )
        self._ttl = ttl_seconds

    def record_transaction(
        self, card_id: str, tx_id: str, amount: float, ts_epoch: float
    ) -> None:
        pipe = self._client.pipeline()
        pipe.zadd(self._card_key(card_id), {tx_id: ts_epoch})
        pipe.set(self._tx_key(tx_id), amount, ex=self._ttl)
        pipe.execute()

    def card_features(
        self, card_id: str, window_seconds: int, now: float | None = None
    ) -> dict:
        """Aggregate features for a card over the trailing window."""
        now = now if now is not None else time.time()
        members = self._client.zrangebyscore(
            self._card_key(card_id), now - window_seconds, now
        )
        if not members:
            return {"tx_count": 0, "tx_total": 0.0, "tx_avg": 0.0}
        amounts = [
            float(value)
            for value in self._client.mget([self._tx_key(m) for m in members])
            if value is not None
        ]
        total = sum(amounts)
        count = len(amounts)
        return {
            "tx_count": count,
            "tx_total": total,
            "tx_avg": round(total / count, 2) if count else 0.0,
        }

    def push_alert(self, alert: dict, maxlen: int = 100) -> None:
        """Store a fraud alert in the recent-alerts list, kept to maxlen items."""
        self._client.lpush(_ALERTS_KEY, json.dumps(alert, default=str))
        self._client.ltrim(_ALERTS_KEY, 0, maxlen - 1)

    def list_alerts(self, limit: int = 100) -> list:
        """Return the most recent fraud alerts, newest first."""
        raw = self._client.lrange(_ALERTS_KEY, 0, limit - 1)
        return [json.loads(item) for item in raw]

    def _card_key(self, card_id: str) -> str:
        return f"fraud:card:{card_id}:txs"

    def _tx_key(self, tx_id: str) -> str:
        return f"fraud:tx:{tx_id}"
