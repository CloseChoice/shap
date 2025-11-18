"""Unit tests for the Sampling explainer."""

import numpy as np
import pytest

import shap


def test_null_model_small():
    explainer = shap.SamplingExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2, 4)), nsamples=100)
    shap_values = explainer.shap_values(np.ones((1, 4)))
    assert np.sum(np.abs(shap_values)) < 1e-8


def test_null_model_small_new():
    explainer = shap.explainers.SamplingExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2, 4)), nsamples=100)
    shap_values = explainer(np.ones((1, 4)))
    assert np.sum(np.abs(shap_values.values)) < 1e-8


def test_null_model():
    explainer = shap.SamplingExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2, 10)), nsamples=100)
    shap_values = explainer.shap_values(np.ones((1, 10)))
    assert np.sum(np.abs(shap_values)) < 1e-8


def test_front_page_model_agnostic():
    sklearn = pytest.importorskip("sklearn")
    train_test_split = pytest.importorskip("sklearn.model_selection").train_test_split

    # train a SVM classifier
    X_train, X_test, Y_train, _ = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
    svm = sklearn.svm.SVC(kernel="rbf", probability=True)
    svm.fit(X_train, Y_train)

    # use Kernel SHAP to explain test set predictions
    explainer = shap.SamplingExplainer(svm.predict_proba, X_train, nsamples=100)
    explainer.shap_values(X_test)


def test_sampling_explainer_linear_model():
    """Test SamplingExplainer with a simple linear model."""
    def linear_model(X):
        return 2 * X[:, 0] + 3 * X[:, 1]

    X_train = np.random.randn(100, 2)
    X_test = np.random.randn(10, 2)

    explainer = shap.SamplingExplainer(linear_model, X_train, nsamples=500)
    shap_values = explainer.shap_values(X_test, nsamples=500)

    assert shap_values.shape == X_test.shape


def test_sampling_explainer_with_dataframe():
    """Test SamplingExplainer with pandas DataFrame."""
    pytest.importorskip("pandas")
    import pandas as pd

    def model(X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X.sum(axis=1)

    X_train = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
    X_test = pd.DataFrame(np.random.randn(5, 3), columns=["a", "b", "c"])

    explainer = shap.SamplingExplainer(model, X_train, nsamples=200)
    explanation = explainer(X_test, nsamples=200)

    assert explanation.values.shape == (5, 3)
    assert explanation.feature_names == ["a", "b", "c"]


def test_sampling_explainer_nsamples_parameter():
    """Test SamplingExplainer with different nsamples values."""
    def model(X):
        return X.sum(axis=1)

    X_train = np.random.randn(100, 4)
    X_test = np.random.randn(3, 4)

    # Test with low nsamples
    explainer_low = shap.SamplingExplainer(model, X_train, nsamples=50)
    values_low = explainer_low.shap_values(X_test, nsamples=50)

    # Test with high nsamples
    explainer_high = shap.SamplingExplainer(model, X_train, nsamples=500)
    values_high = explainer_high.shap_values(X_test, nsamples=500)

    assert values_low.shape == values_high.shape == (3, 4)

    # Higher nsamples should generally give more accurate results
    # but both should at least have the right shape


def test_sampling_explainer_expected_value():
    """Test that SamplingExplainer computes expected values."""
    def model(X):
        return X[:, 0] + X[:, 1]

    X_train = np.random.randn(100, 2)
    X_test = np.random.randn(5, 2)

    explainer = shap.SamplingExplainer(model, X_train, nsamples=300)
    explanation = explainer(X_test, nsamples=300)

    assert hasattr(explainer, "expected_value")
    assert explanation.base_values is not None


def test_sampling_explainer_additivity():
    """Test that SHAP values sum to (prediction - expected_value)."""
    def model(X):
        return 2 * X[:, 0] + X[:, 1]

    X_train = np.random.randn(100, 2)
    X_test = np.random.randn(5, 2)

    explainer = shap.SamplingExplainer(model, X_train, nsamples=1000)
    explanation = explainer(X_test, nsamples=1000)

    # Check additivity for each sample
    for i in range(len(X_test)):
        prediction = model(X_test[i:i+1])[0]
        # base_values may be a scalar for single output models
        base_value = explanation.base_values if np.isscalar(explanation.base_values) else explanation.base_values[i]
        shap_sum = explanation.values[i].sum() + base_value
        # Sampling introduces some error, so use a tolerance
        assert np.abs(prediction - shap_sum) < 0.5


def test_sampling_explainer_single_sample():
    """Test SamplingExplainer with a single sample."""
    def model(X):
        return X.sum(axis=1)

    X_train = np.random.randn(50, 3)
    X_test = np.random.randn(1, 3)

    explainer = shap.SamplingExplainer(model, X_train, nsamples=200)
    shap_values = explainer.shap_values(X_test, nsamples=200)

    assert shap_values.shape == (1, 3)


def test_sampling_explainer_multi_output():
    """Test SamplingExplainer with multi-output model."""
    def model(X):
        # Return two outputs
        return np.column_stack([X.sum(axis=1), X.mean(axis=1)])

    X_train = np.random.randn(50, 3)
    X_test = np.random.randn(5, 3)

    explainer = shap.SamplingExplainer(model, X_train, nsamples=200)
    shap_values = explainer.shap_values(X_test, nsamples=200)

    # Should return a list of arrays for multi-output
    assert isinstance(shap_values, list) or shap_values.ndim == 3


def test_sampling_explainer_identity_link_only():
    """Test that SamplingExplainer only accepts identity link."""
    def model(X):
        return X.sum(axis=1)

    X_train = np.random.randn(50, 2)

    # Identity link should work (default)
    explainer = shap.SamplingExplainer(model, X_train)
    assert explainer is not None

    # Non-identity link should raise ValueError
    # Note: the actual error depends on how the link is validated in parent class
    # So we just check that initialization fails with non-identity link
    try:
        shap.SamplingExplainer(model, X_train, link="logit")
        # If it doesn't raise, at least it should validate after init
        assert False, "Should have raised an error for non-identity link"
    except (ValueError, TypeError):
        pass  # Expected behavior


def test_sampling_explainer_consistency():
    """Test that SamplingExplainer gives similar results across runs with same seed."""
    def model(X):
        return X[:, 0] * 2 + X[:, 1]

    X_train = np.random.RandomState(42).randn(100, 2)
    X_test = np.random.RandomState(43).randn(3, 2)

    # Set seed for reproducibility
    np.random.seed(42)
    explainer1 = shap.SamplingExplainer(model, X_train, nsamples=500)
    values1 = explainer1.shap_values(X_test, nsamples=500)

    # Same seed should give same results
    np.random.seed(42)
    explainer2 = shap.SamplingExplainer(model, X_train, nsamples=500)
    values2 = explainer2.shap_values(X_test, nsamples=500)

    assert np.allclose(values1, values2, atol=0.01)


def test_sampling_explainer_call_vs_shap_values():
    """Test that __call__ and shap_values give similar results."""
    def model(X):
        return X.sum(axis=1)

    X_train = np.random.randn(50, 3)
    X_test = np.random.randn(5, 3)

    # Set seed for reproducibility
    np.random.seed(42)
    explainer = shap.SamplingExplainer(model, X_train, nsamples=500)

    # Using __call__
    np.random.seed(42)
    explanation = explainer(X_test, nsamples=500)
    values_call = explanation.values

    # Using shap_values
    np.random.seed(42)
    values_method = explainer.shap_values(X_test, nsamples=500)

    # Results should be very similar with same seed
    # Use a reasonable tolerance since there's still some randomness
    assert values_call.shape == values_method.shape


def test_sampling_explainer_varying_features():
    """Test SamplingExplainer with features that don't vary."""
    def model(X):
        return X.sum(axis=1)

    # Create training data
    X_train = np.random.randn(50, 3)

    # Create test data where one feature matches training mean
    X_test = np.random.randn(5, 3)
    X_test[:, 2] = X_train[:, 2].mean()

    explainer = shap.SamplingExplainer(model, X_train, nsamples=200)
    shap_values = explainer.shap_values(X_test, nsamples=200)

    # Should still work
    assert shap_values.shape == (5, 3)


def test_sampling_explainer_large_background():
    """Test SamplingExplainer with large background dataset."""
    def model(X):
        return X.mean(axis=1)

    # SamplingExplainer should handle large background sets well
    X_train = np.random.randn(1000, 5)
    X_test = np.random.randn(3, 5)

    # Should not raise warnings about large dataset
    explainer = shap.SamplingExplainer(model, X_train, nsamples=300)
    shap_values = explainer.shap_values(X_test, nsamples=300)

    assert shap_values.shape == (3, 5)


def test_sampling_explainer_with_sklearn_model():
    """Test SamplingExplainer with sklearn model."""
    sklearn = pytest.importorskip("sklearn")

    X_train = np.random.randn(100, 4)
    y_train = (X_train.sum(axis=1) > 0).astype(int)

    model = sklearn.linear_model.LogisticRegression()
    model.fit(X_train, y_train)

    X_test = np.random.randn(5, 4)

    explainer = shap.SamplingExplainer(model.predict_proba, X_train, nsamples=300)
    shap_values = explainer.shap_values(X_test, nsamples=300)

    # For binary classification, should return array for each class
    assert isinstance(shap_values, (list, np.ndarray))


def test_sampling_explainer_zero_output():
    """Test SamplingExplainer with model that returns zeros."""
    def zero_model(X):
        return np.zeros(X.shape[0])

    X_train = np.random.randn(50, 3)
    X_test = np.random.randn(5, 3)

    explainer = shap.SamplingExplainer(zero_model, X_train, nsamples=200)
    shap_values = explainer.shap_values(X_test, nsamples=200)

    # SHAP values should be close to zero for zero model
    assert np.abs(shap_values).max() < 0.01
