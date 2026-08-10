"""Pluggable classifier registry shared by the training jobs.

Register a new model by adding it to CLASSIFIERS; training selects it via
MODEL_TYPE / REAL_MODEL_TYPE and scoring works unchanged because every
estimator is wrapped as an MLflow pyfunc returning probabilities.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

CLASSIFIERS = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42
    ),
}


def build_classifier(model_type: str):
    """Instantiate a fresh classifier for the given model type."""
    try:
        return CLASSIFIERS[model_type]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown model type '{model_type}'. Available: {sorted(CLASSIFIERS)}"
        ) from exc
