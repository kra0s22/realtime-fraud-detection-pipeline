"""Feature engineering shared by training and online inference."""

import pandas as pd

from producer.generator import TransactionGenerator

# Exact column order the inference API sends to the model.
FEATURE_COLUMNS = ["amount", "channel", "country", "tx_count", "tx_total", "tx_avg"]


def build_training_frame(
    n_transactions: int = 20000,
    fraud_rate: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a labeled training frame with per-card velocity features."""
    generator = TransactionGenerator(fraud_rate=fraud_rate, seed=seed)
    records = [
        tx.model_dump() for tx in (generator.generate() for _ in range(n_transactions))
    ]
    frame = pd.DataFrame(records)

    card_stats = (
        frame.groupby("card_id")
        .agg(
            tx_count=("transaction_id", "count"),
            tx_total=("amount", "sum"),
        )
        .reset_index()
    )
    card_stats["tx_avg"] = card_stats["tx_total"] / card_stats["tx_count"]
    frame = frame.merge(card_stats, on="card_id", how="left")
    frame["tx_count"] = frame["tx_count"].astype(int)
    frame["is_fraud"] = frame["is_fraud"].astype(int)
    return frame[FEATURE_COLUMNS + ["is_fraud"]]
