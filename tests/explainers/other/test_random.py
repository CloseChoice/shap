"""Tests for Random explainer."""

import numpy as np
import pytest
import sklearn.ensemble
import sklearn.linear_model

import shap


def test_random_explainer_basic():
    """Test basic functionality of Random explainer."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    # Create masker
    masker = shap.maskers.Independent(X_train)

    # Create explainer
    explainer = shap.explainers.other.Random(model.predict, masker)

    # Get explanation
    explanation = explainer(X_test[:5])

    # Check that explanation has values
    assert explanation.values is not None
    assert explanation.values.shape[0] == 5  # 5 samples
    assert explanation.values.shape[1] == X_test.shape[1]  # number of features

    # Check that base values are present
    assert explanation.base_values is not None
    assert len(explanation.base_values) == 5


def test_random_explainer_values_are_small():
    """Test that Random explainer produces small values (as per the implementation)."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.other.Random(model.predict, masker)

    explanation = explainer(X_test[:10])

    # Values should be small (multiplied by 0.001 in the implementation)
    assert np.abs(explanation.values).max() < 0.1  # Should be very small


def test_random_explainer_different_samples():
    """Test that Random explainer produces different values for different samples (non-constant mode)."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.other.Random(model.predict, masker, constant=False)

    explanation = explainer(X_test[:5])

    # Values should be different for different samples
    # (though there's a small chance they could be similar due to randomness)
    values_vary = False
    for i in range(explanation.values.shape[1]):
        if not np.allclose(explanation.values[:, i], explanation.values[0, i], atol=1e-6):
            values_vary = True
            break

    # At least some features should vary across samples
    assert values_vary or explanation.values.shape[0] == 1


def test_random_explainer_with_tree_model():
    """Test Random explainer with a tree-based model."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.ensemble.RandomForestRegressor(n_estimators=10, max_depth=5, random_state=0)
    model.fit(X_train, y_train)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.other.Random(model.predict, masker)

    explanation = explainer(X_test[:5])

    # Check basic properties
    assert explanation.values.shape == (5, X_test.shape[1])
    assert len(explanation.base_values) == 5


def test_random_explainer_with_identity_link():
    """Test Random explainer with identity link function."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.other.Random(model.predict, masker, link=shap.links.identity)

    explanation = explainer(X_test[:5])

    assert explanation.values is not None
    assert explanation.values.shape == (5, X_test.shape[1])


def test_random_explainer_consistency_with_seed():
    """Test that Random explainer produces consistent results when numpy seed is set."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    masker = shap.maskers.Independent(X_train)

    # Set seed and get explanation
    np.random.seed(42)
    explainer1 = shap.explainers.other.Random(model.predict, masker)
    explanation1 = explainer1(X_test[:5])

    # Set seed again and get explanation
    np.random.seed(42)
    explainer2 = shap.explainers.other.Random(model.predict, masker)
    explanation2 = explainer2(X_test[:5])

    # Results should be the same with same seed
    assert np.allclose(explanation1.values, explanation2.values)


def test_random_explainer_with_single_sample():
    """Test Random explainer with a single sample."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.other.Random(model.predict, masker)

    # Test with single sample
    single_sample = X_test[:1]
    explanation = explainer(single_sample)

    assert explanation.values.shape == (1, X_test.shape[1])
    assert len(explanation.base_values) == 1


def test_random_explainer_expected_value():
    """Test that Random explainer computes expected values correctly."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.other.Random(model.predict, masker)

    explanation = explainer(X_test[:5])

    # Base values should exist
    assert explanation.base_values is not None
    assert len(explanation.base_values) == 5


def test_random_explainer_output_shape():
    """Test that Random explainer output has correct shape."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.other.Random(model.predict, masker)

    # Test with different numbers of samples
    for n_samples in [1, 5, 10]:
        explanation = explainer(X_test[:n_samples])
        assert explanation.values.shape == (n_samples, X_test.shape[1])
        assert len(explanation.base_values) == n_samples


def test_random_explainer_with_classifier():
    """Test Random explainer with a classification model."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_classification(n_samples=100, n_features=8, random_state=0), test_size=0.1, random_state=0
    )

    model = sklearn.linear_model.LogisticRegression(solver="liblinear", random_state=0)
    model.fit(X_train, y_train)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.other.Random(model.predict_proba, masker)

    explanation = explainer(X_test[:5])

    # For binary classification with predict_proba, output should have shape for both classes
    assert explanation.values is not None
    # Values shape should match features
    assert explanation.values.shape[0] == 5
    assert explanation.values.shape[1] == X_test.shape[1]


def test_random_explainer_with_custom_max_evals():
    """Test Random explainer with custom max_evals parameter."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.other.Random(model.predict, masker)

    # max_evals doesn't really matter for Random explainer, but test it works
    explanation = explainer(X_test[:5], max_evals=100)

    assert explanation.values is not None
    assert explanation.values.shape == (5, X_test.shape[1])


def test_random_explainer_linearize_link():
    """Test Random explainer with linearize_link parameter."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    masker = shap.maskers.Independent(X_train)

    # Test with linearize_link=True
    explainer_true = shap.explainers.other.Random(model.predict, masker, linearize_link=True)
    explanation_true = explainer_true(X_test[:5])

    # Test with linearize_link=False
    explainer_false = shap.explainers.other.Random(model.predict, masker, linearize_link=False)
    explanation_false = explainer_false(X_test[:5])

    # Both should produce valid explanations
    assert explanation_true.values is not None
    assert explanation_false.values is not None


def test_random_explainer_benchmark_purpose():
    """Test that Random explainer is suitable for benchmarking (produces consistent small random values)."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    masker = shap.maskers.Independent(X_train)

    np.random.seed(42)
    explainer = shap.explainers.other.Random(model.predict, masker)
    explanation = explainer(X_test[:10])

    # Check that values are random but small (good baseline for benchmarking)
    assert explanation.values.std() > 0  # Should have some variance
    assert np.abs(explanation.values).max() < 0.1  # But should be small

    # Check that the explanation structure is complete
    assert hasattr(explanation, "values")
    assert hasattr(explanation, "base_values")
    assert hasattr(explanation, "data")
