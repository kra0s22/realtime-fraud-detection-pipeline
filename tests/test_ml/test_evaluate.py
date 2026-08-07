"""Unit tests for model evaluation metrics."""

import pytest

pytest.importorskip("sklearn")

from sklearn.datasets import make_classification  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from ml.evaluate import evaluate  # noqa: E402


def test_evaluate_returns_expected_metrics():
    x, y = make_classification(
        n_samples=300, n_features=6, n_informative=4, random_state=1
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(x, y)
    metrics = evaluate(model, x, y)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
