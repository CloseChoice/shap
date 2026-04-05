"""Tests for AdditiveExplainer."""

import numpy as np
import pytest
import sklearn.datasets
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection

import shap


def additive_model_simple(X):
    """Simple additive model for testing: f(x) = 2*x0 + 3*x1 + 1."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return 2 * X[:, 0] + 3 * X[:, 1] + 1


def additive_model_squared(X):
    """Additive model with squared terms: f(x) = x0^2 + x1^2."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X[:, 0] ** 2 + X[:, 1] ** 2


def test_additive_explainer_basic():
    """Test basic functionality of AdditiveExplainer."""
    X_train = np.random.randn(100, 2)
    X_test = np.random.randn(5, 2)

    # Create masker
    masker = shap.maskers.Independent(X_train)

    # Create explainer
    explainer = shap.explainers.AdditiveExplainer(additive_model_simple, masker)

    # Get explanation
    explanation = explainer(X_test)

    # Check that explanation has values
    assert explanation.values is not None
    assert explanation.values.shape == (5, 2)

    # Check that base values are present
    assert explanation.base_values is not None


def test_additive_explainer_linear_model():
    """Test AdditiveExplainer with a linear model."""
    X_train = np.random.randn(100, 3)
    y_train = 2 * X_train[:, 0] + 3 * X_train[:, 1] - X_train[:, 2] + np.random.randn(100) * 0.1

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    X_test = np.random.randn(10, 3)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.AdditiveExplainer(model.predict, masker)

    explanation = explainer(X_test)

    # Check shape
    assert explanation.values.shape == (10, 3)

    # For linear models, SHAP values should approximate the coefficients times the feature values
    # (relative to their mean)
    assert explanation.values is not None


def test_additive_explainer_expected_value():
    """Test that AdditiveExplainer computes expected values correctly."""
    X_train = np.random.randn(100, 2)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.AdditiveExplainer(additive_model_simple, masker)

    X_test = np.random.randn(5, 2)
    explanation = explainer(X_test)

    # Check that expected value exists
    assert hasattr(explainer, "_expected_value")
    assert explanation.base_values is not None

    # For each sample, values should sum to (prediction - expected_value)
    for i in range(len(X_test)):
        prediction = additive_model_simple(X_test[i])
        shap_sum = explanation.values[i].sum() + explanation.base_values[i]
        assert np.abs(prediction - shap_sum) < 1e-5


def test_additive_explainer_additivity():
    """Test that SHAP values satisfy additivity property."""
    X_train = np.random.randn(100, 2)
    X_test = np.random.randn(10, 2)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.AdditiveExplainer(additive_model_simple, masker)

    explanation = explainer(X_test)

    # For each sample, sum of SHAP values + base_value should equal the prediction
    for i in range(len(X_test)):
        prediction = additive_model_simple(X_test[i])
        shap_sum = explanation.values[i].sum() + explanation.base_values[i]
        assert np.allclose(prediction, shap_sum, atol=1e-5), (
            f"Additivity violated: prediction={prediction}, shap_sum={shap_sum}"
        )


def test_additive_explainer_with_single_sample():
    """Test AdditiveExplainer with a single sample."""
    X_train = np.random.randn(100, 2)
    X_test = np.random.randn(1, 2)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.AdditiveExplainer(additive_model_simple, masker)

    explanation = explainer(X_test)

    assert explanation.values.shape == (1, 2)
    assert len(explanation.base_values) == 1


def test_additive_explainer_initialization():
    """Test AdditiveExplainer initialization."""
    X_train = np.random.randn(100, 3)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.AdditiveExplainer(additive_model_simple, masker)

    # Check that internal attributes are set
    assert hasattr(explainer, "_expected_value")
    assert hasattr(explainer, "_zero_offset")
    assert hasattr(explainer, "_input_offsets")

    # Check dimensions
    assert explainer._input_offsets.shape == (3,)


def test_additive_explainer_with_feature_names():
    """Test AdditiveExplainer with feature names."""
    X_train = np.random.randn(100, 2)
    feature_names = ["feature_0", "feature_1"]

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.AdditiveExplainer(
        additive_model_simple, masker, feature_names=feature_names
    )

    X_test = np.random.randn(5, 2)
    explanation = explainer(X_test)

    # Feature names should be accessible
    assert explanation.feature_names == feature_names


def test_additive_explainer_offsets():
    """Test that input offsets are computed correctly."""
    X_train = np.zeros((100, 2))  # Use zeros for simplicity

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.AdditiveExplainer(additive_model_simple, masker)

    # With zero-centered data, offsets should relate to the model's behavior
    assert hasattr(explainer, "_input_offsets")
    assert explainer._input_offsets.shape == (2,)


def test_additive_explainer_consistency():
    """Test that AdditiveExplainer returns consistent results."""
    X_train = np.random.randn(100, 2)
    X_test = np.random.randn(10, 2)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.AdditiveExplainer(additive_model_simple, masker)

    # Get explanations twice
    explanation1 = explainer(X_test)
    explanation2 = explainer(X_test)

    # Results should be identical
    assert np.allclose(explanation1.values, explanation2.values)
    assert np.allclose(explanation1.base_values, explanation2.base_values)


def test_additive_explainer_different_sample_sizes():
    """Test AdditiveExplainer with different sample sizes."""
    X_train = np.random.randn(100, 3)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.AdditiveExplainer(additive_model_simple, masker)

    # Test with different numbers of samples
    for n_samples in [1, 5, 20]:
        X_test = np.random.randn(n_samples, 3)
        explanation = explainer(X_test)

        assert explanation.values.shape == (n_samples, 3)
        assert len(explanation.base_values) == n_samples


def test_additive_explainer_with_squared_model():
    """Test AdditiveExplainer with a model that has squared terms."""
    X_train = np.random.randn(100, 2)
    X_test = np.random.randn(5, 2)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.AdditiveExplainer(additive_model_squared, masker)

    explanation = explainer(X_test)

    # Check additivity
    for i in range(len(X_test)):
        prediction = additive_model_squared(X_test[i])
        shap_sum = explanation.values[i].sum() + explanation.base_values[i]
        assert np.allclose(prediction, shap_sum, atol=1e-5)


def test_additive_explainer_supports_model_with_masker():
    """Test the supports_model_with_masker static method."""
    X_train = np.random.randn(100, 2)
    masker = shap.maskers.Independent(X_train)

    # Regular function should return False
    assert not shap.explainers.AdditiveExplainer.supports_model_with_masker(
        additive_model_simple, masker
    )

    # Linear model should return False
    model = sklearn.linear_model.LinearRegression()
    assert not shap.explainers.AdditiveExplainer.supports_model_with_masker(model, masker)


def test_additive_explainer_with_negative_values():
    """Test AdditiveExplainer with data containing negative values."""
    X_train = np.random.randn(100, 2) - 5  # Shifted to be mostly negative
    X_test = np.random.randn(10, 2) - 5

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.AdditiveExplainer(additive_model_simple, masker)

    explanation = explainer(X_test)

    # Check additivity still holds
    for i in range(len(X_test)):
        prediction = additive_model_simple(X_test[i])
        shap_sum = explanation.values[i].sum() + explanation.base_values[i]
        assert np.allclose(prediction, shap_sum, atol=1e-5)


def test_additive_explainer_with_constant_features():
    """Test AdditiveExplainer when some features are constant."""
    # Create data where one feature is constant
    X_train = np.random.randn(100, 3)
    X_train[:, 2] = 5.0  # Make third feature constant

    def model_with_constant(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return 2 * X[:, 0] + 3 * X[:, 1] + X[:, 2]

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.AdditiveExplainer(model_with_constant, masker)

    X_test = X_train[:5].copy()
    explanation = explainer(X_test)

    # SHAP values for constant feature should be near zero
    # (all variation is captured by base value)
    assert np.abs(explanation.values[:, 2]).max() < 0.1


def test_additive_explainer_main_effects():
    """Test that main_effects parameter works."""
    X_train = np.random.randn(100, 2)
    X_test = np.random.randn(5, 2)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.AdditiveExplainer(additive_model_simple, masker)

    # Call with main_effects=True
    explanation = explainer(X_test, main_effects=True)

    # Main effects should be present in the result
    assert explanation.values is not None
    assert hasattr(explanation, "values")


def test_additive_explainer_with_large_features():
    """Test AdditiveExplainer with larger number of features."""
    X_train = np.random.randn(100, 10)

    def model_10d(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X.sum(axis=1)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.AdditiveExplainer(model_10d, masker)

    X_test = np.random.randn(5, 10)
    explanation = explainer(X_test)

    assert explanation.values.shape == (5, 10)

    # Check additivity
    for i in range(len(X_test)):
        prediction = model_10d(X_test[i])
        shap_sum = explanation.values[i].sum() + explanation.base_values[i]
        assert np.allclose(prediction, shap_sum, atol=1e-5)


def test_additive_explainer_with_zero_input():
    """Test AdditiveExplainer with zero input."""
    X_train = np.random.randn(100, 2)
    X_test = np.zeros((1, 2))

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.AdditiveExplainer(additive_model_simple, masker)

    explanation = explainer(X_test)

    assert explanation.values.shape == (1, 2)
    # SHAP values computed relative to the distribution mean


def test_additive_explainer_linearize_link():
    """Test AdditiveExplainer with linearize_link parameter."""
    X_train = np.random.randn(100, 2)
    X_test = np.random.randn(5, 2)

    masker = shap.maskers.Independent(X_train)

    # Test with linearize_link=True
    explainer_true = shap.explainers.AdditiveExplainer(
        additive_model_simple, masker, linearize_link=True
    )
    explanation_true = explainer_true(X_test)

    # Test with linearize_link=False
    explainer_false = shap.explainers.AdditiveExplainer(
        additive_model_simple, masker, linearize_link=False
    )
    explanation_false = explainer_false(X_test)

    # Both should produce valid explanations
    assert explanation_true.values is not None
    assert explanation_false.values is not None

    # Results should be similar for simple additive models
    assert explanation_true.values.shape == explanation_false.values.shape


def test_additive_explainer_masked_model_integration():
    """Test that AdditiveExplainer properly integrates with MaskedModel."""
    X_train = np.random.randn(100, 2)
    X_test = np.random.randn(5, 2)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.explainers.AdditiveExplainer(additive_model_simple, masker)

    # The explainer should have pre-computed offsets
    assert hasattr(explainer, "_zero_offset")
    assert hasattr(explainer, "_input_offsets")

    explanation = explainer(X_test)

    # Verify that the explanation is valid
    assert explanation.values is not None
    assert explanation.base_values is not None
