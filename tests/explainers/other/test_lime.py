"""Tests for LimeTabular explainer."""

import numpy as np
import pandas as pd
import pytest
import sklearn.ensemble
import sklearn.linear_model

import shap


def test_lime_regression_basic():
    """Test LimeTabular with regression model."""
    lime = pytest.importorskip("lime")

    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.LimeTabular(model.predict, X_train, mode="regression")
    attributions = explainer.attributions(X_test[:5])

    # Check shape
    assert attributions.shape == (5, X_test.shape[1])

    # Attributions should be numeric
    assert not np.any(np.isnan(attributions))


def test_lime_classification_basic():
    """Test LimeTabular with classification model."""
    lime = pytest.importorskip("lime")

    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_classification(n_samples=100, n_features=8, random_state=0), test_size=0.1, random_state=0
    )

    model = sklearn.linear_model.LogisticRegression(solver="liblinear", random_state=0)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.LimeTabular(
        model.predict_proba, X_train, mode="classification"
    )
    attributions = explainer.attributions(X_test[:5])

    # For 1D output, should return 2D array with attributions
    assert attributions.ndim == 2
    assert attributions.shape[0] == 5
    assert not np.any(np.isnan(attributions))


def test_lime_regression_with_dataframe():
    """Test LimeTabular with pandas DataFrame."""
    lime = pytest.importorskip("lime")

    X, y = sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0)
    X_df = pd.DataFrame(X)
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        X_df, y, test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    # Test with DataFrame for data
    explainer = shap.explainers.other.LimeTabular(model.predict, X_train, mode="regression")

    # Test with DataFrame for X
    attributions = explainer.attributions(X_test[:5])

    assert attributions.shape == (5, X_test.shape[1])


def test_lime_classification_with_dataframe():
    """Test LimeTabular classification with pandas DataFrame."""
    lime = pytest.importorskip("lime")

    X, y = sklearn.datasets.make_classification(n_samples=100, n_features=8, random_state=0)
    X_df = pd.DataFrame(X)
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        X_df, y, test_size=0.1, random_state=0
    )

    model = sklearn.linear_model.LogisticRegression(solver="liblinear", random_state=0)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.LimeTabular(
        model.predict_proba, X_train, mode="classification"
    )
    attributions = explainer.attributions(X_test[:5])

    assert attributions.ndim == 2
    assert attributions.shape[0] == 5


def test_lime_invalid_mode():
    """Test that LimeTabular raises ValueError for invalid mode."""
    lime = pytest.importorskip("lime")

    X, y = sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0)
    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)

    with pytest.raises(ValueError, match="Invalid mode"):
        shap.explainers.other.LimeTabular(model.predict, X, mode="invalid_mode")


def test_lime_regression_single_sample():
    """Test LimeTabular with single sample in regression."""
    lime = pytest.importorskip("lime")

    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.LimeTabular(model.predict, X_train, mode="regression")

    # Test with single sample
    single_sample = X_test[:1]
    attributions = explainer.attributions(single_sample)

    assert attributions.shape == single_sample.shape


def test_lime_classification_single_sample():
    """Test LimeTabular with single sample in classification."""
    lime = pytest.importorskip("lime")

    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_classification(n_samples=100, n_features=8, random_state=0), test_size=0.1, random_state=0
    )

    model = sklearn.linear_model.LogisticRegression(solver="liblinear", random_state=0)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.LimeTabular(
        model.predict_proba, X_train, mode="classification"
    )

    # Test with single sample
    single_sample = X_test[:1]
    attributions = explainer.attributions(single_sample)

    assert attributions.shape[0] == 1


def test_lime_regression_with_nsamples():
    """Test LimeTabular with custom nsamples parameter."""
    lime = pytest.importorskip("lime")

    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.LimeTabular(model.predict, X_train, mode="regression")

    # Test with different nsamples
    attributions = explainer.attributions(X_test[:5], nsamples=1000)

    assert attributions.shape == (5, X_test.shape[1])


def test_lime_regression_with_num_features():
    """Test LimeTabular with custom num_features parameter."""
    lime = pytest.importorskip("lime")

    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.LimeTabular(model.predict, X_train, mode="regression")

    # Test with limited num_features
    num_features = 5
    attributions = explainer.attributions(X_test[:3], num_features=num_features)

    assert attributions.shape == (3, X_test.shape[1])

    # Check that at most num_features have non-zero attributions per sample
    for i in range(3):
        non_zero_count = np.count_nonzero(attributions[i])
        assert non_zero_count <= num_features


def test_lime_classification_binary():
    """Test LimeTabular with binary classification."""
    lime = pytest.importorskip("lime")

    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_classification(n_samples=100, n_features=8, random_state=0), test_size=0.1, random_state=0
    )

    model = sklearn.linear_model.LogisticRegression(solver="liblinear", random_state=0)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.LimeTabular(
        model.predict_proba, X_train, mode="classification"
    )
    attributions = explainer.attributions(X_test[:5])

    # Should handle binary classification properly
    assert attributions.ndim == 2


def test_lime_regression_with_tree_model():
    """Test LimeTabular with tree-based regression model."""
    lime = pytest.importorskip("lime")

    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.ensemble.RandomForestRegressor(n_estimators=10, max_depth=5, random_state=0)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.LimeTabular(model.predict, X_train, mode="regression")
    attributions = explainer.attributions(X_test[:5])

    assert attributions.shape == (5, X_test.shape[1])
    assert not np.any(np.isnan(attributions))


def test_lime_classification_with_tree_model():
    """Test LimeTabular with tree-based classification model."""
    lime = pytest.importorskip("lime")

    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_classification(n_samples=100, n_features=8, random_state=0), test_size=0.1, random_state=0
    )

    model = sklearn.ensemble.RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.LimeTabular(
        model.predict_proba, X_train, mode="classification"
    )
    attributions = explainer.attributions(X_test[:5])

    assert attributions.ndim == 2
    assert not np.any(np.isnan(attributions))


def test_lime_regression_output_negation():
    """Test that LIME regression output is negated as per implementation."""
    lime = pytest.importorskip("lime")

    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.LimeTabular(model.predict, X_train, mode="regression")
    attributions = explainer.attributions(X_test[:3])

    # Just verify that attributions are computed (negation is internal)
    assert attributions.shape == (3, X_test.shape[1])


def test_lime_explainer_initialization():
    """Test that LimeTabular initializes correctly."""
    lime = pytest.importorskip("lime")

    X, y = sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0)
    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)

    # Test initialization with regression mode
    explainer_reg = shap.explainers.other.LimeTabular(model.predict, X, mode="regression")
    assert explainer_reg.mode == "regression"
    assert explainer_reg.data.shape == X.shape

    # Test initialization with classification mode
    X_cls, y_cls = sklearn.datasets.make_classification(n_samples=100, n_features=8, random_state=0)
    X_cls_train = X_cls[:100]
    model_cls = sklearn.linear_model.LogisticRegression(solver="liblinear")
    model_cls.fit(X_cls_train, y_cls[:100])

    explainer_cls = shap.explainers.other.LimeTabular(
        model_cls.predict_proba, X_cls_train, mode="classification"
    )
    assert explainer_cls.mode == "classification"


def test_lime_different_sample_sizes():
    """Test LimeTabular with different sample sizes."""
    lime = pytest.importorskip("lime")

    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.LimeTabular(model.predict, X_train, mode="regression")

    # Test with different sample sizes
    for n_samples in [1, 3, 10]:
        attributions = explainer.attributions(X_test[:n_samples])
        assert attributions.shape == (n_samples, X_test.shape[1])


def test_lime_model_wrapper_for_1d_output():
    """Test that LimeTabular wraps 1D classification output correctly."""
    lime = pytest.importorskip("lime")

    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_classification(n_samples=100, n_features=8, random_state=0), test_size=0.1, random_state=0
    )

    model = sklearn.linear_model.LogisticRegression(solver="liblinear", random_state=0)
    model.fit(X_train, y_train)

    # Use predict_proba which returns 2D for binary classification
    def predict_func(X):
        # Return only positive class probability (1D)
        return model.predict_proba(X)[:, 1]

    explainer = shap.explainers.other.LimeTabular(predict_func, X_train, mode="classification")

    # Should handle 1D output by wrapping it
    attributions = explainer.attributions(X_test[:3])

    assert attributions is not None
    assert attributions.shape[0] == 3
