"""Unit tests for the pluggable classifier registry."""

import pytest

pytest.importorskip("sklearn")

from ml.models import CLASSIFIERS, build_classifier  # noqa: E402


def test_registry_exposes_expected_models():
    assert "logistic_regression" in CLASSIFIERS
    assert "random_forest" in CLASSIFIERS


def test_build_classifier_returns_registered_types():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    assert isinstance(build_classifier("logistic_regression"), LogisticRegression)
    assert isinstance(build_classifier("random_forest"), RandomForestClassifier)


def test_build_classifier_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown model type"):
        build_classifier("does-not-exist")
