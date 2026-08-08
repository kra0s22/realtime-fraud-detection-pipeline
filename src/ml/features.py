"""Feature engineering shared by training and online inference."""

import logging

import pandas as pd

from producer.generator import TransactionGenerator

logger = logging.getLogger("ml.features")

# Exact column order the inference API sends to the model.
FEATURE_COLUMNS = ["amount", "channel", "country", "tx_count", "tx_total", "tx_avg"]


def build_training_frame(
    n_transactions: int = 20000,
    fraud_rate: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a labeled training frame with per-card velocity features (synthetic)."""
    generator = TransactionGenerator(fraud_rate=fraud_rate, seed=seed)
    records = [
        tx.model_dump() for tx in (generator.generate() for _ in range(n_transactions))
    ]
    return _with_card_features(pd.DataFrame(records))


def frame_from_delta(delta_table_path: str) -> pd.DataFrame:
    """Read accumulated transactions from Delta Lake and build the training frame."""
    from deltalake import DeltaTable

    frame = DeltaTable(delta_table_path).to_pandas()
    if frame.empty:
        raise ValueError("Delta table is empty")
    frame["amount"] = frame["amount"].astype(float)
    frame["is_fraud"] = frame["is_fraud"].astype(int)
    return _with_card_features(frame)


def _with_card_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach per-card velocity features and select the canonical columns."""
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
    return frame[FEATURE_COLUMNS + ["is_fraud"]]
