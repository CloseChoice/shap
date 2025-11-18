"""Tests for Coefficient explainer."""

import numpy as np
import pytest
import sklearn.linear_model

import shap


def test_coefficient_linear_regression():
    """Test Coefficient explainer with LinearRegression."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.Coefficient(model)
    attributions = explainer.attributions(X_test)

    # Check shape
    assert attributions.shape == X_test.shape

    # Check that all rows are identical (same coefficients)
    assert np.allclose(attributions[0], attributions[1])

    # Check that attributions match model's coefficients
    assert np.allclose(attributions[0], model.coef_)


def test_coefficient_ridge_regression():
    """Test Coefficient explainer with Ridge regression."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.Ridge(alpha=1.0, random_state=0)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.Coefficient(model)
    attributions = explainer.attributions(X_test)

    # Check shape
    assert attributions.shape == X_test.shape

    # Check that attributions match model's coefficients
    assert np.allclose(attributions[0], model.coef_)


def test_coefficient_lasso_regression():
    """Test Coefficient explainer with Lasso regression."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.Lasso(alpha=0.1, random_state=0, max_iter=2000)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.Coefficient(model)
    attributions = explainer.attributions(X_test)

    # Check shape
    assert attributions.shape == X_test.shape

    # Check that attributions match model's coefficients
    assert np.allclose(attributions[0], model.coef_)


def test_coefficient_elastic_net():
    """Test Coefficient explainer with ElasticNet."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.ElasticNet(alpha=0.1, random_state=0, max_iter=2000)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.Coefficient(model)
    attributions = explainer.attributions(X_test)

    # Check shape
    assert attributions.shape == X_test.shape

    # Check that attributions match model's coefficients
    assert np.allclose(attributions[0], model.coef_)


def test_coefficient_logistic_regression():
    """Test Coefficient explainer with LogisticRegression."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_classification(n_samples=100, n_features=8, random_state=0), test_size=0.1, random_state=0
    )

    model = sklearn.linear_model.LogisticRegression(solver="liblinear", random_state=0)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.Coefficient(model)
    attributions = explainer.attributions(X_test)

    # Check shape
    assert attributions.shape == X_test.shape

    # For binary classification, coef_ is 2D but we need to handle it
    # Check that all rows are identical
    assert np.allclose(attributions[0], attributions[-1])


def test_coefficient_sgd_regressor():
    """Test Coefficient explainer with SGDRegressor."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.SGDRegressor(random_state=0, max_iter=1000)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.Coefficient(model)
    attributions = explainer.attributions(X_test)

    # Check shape
    assert attributions.shape == X_test.shape

    # Check that attributions match model's coefficients
    assert np.allclose(attributions[0], model.coef_)


def test_coefficient_model_without_coef():
    """Test that Coefficient raises AssertionError for models without coef_ attribute."""
    from sklearn.tree import DecisionTreeRegressor

    X, y = sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0)
    model = DecisionTreeRegressor()
    model.fit(X, y)

    with pytest.raises(AssertionError, match="does not have a coef_"):
        shap.explainers.other.Coefficient(model)


def test_coefficient_with_single_sample():
    """Test Coefficient explainer with a single sample."""
    X_train, _, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.Coefficient(model)

    # Test with single sample
    single_sample = X_train[:1]
    attributions = explainer.attributions(single_sample)

    assert attributions.shape == single_sample.shape
    assert np.allclose(attributions[0], model.coef_)


def test_coefficient_consistency_across_calls():
    """Test that Coefficient returns consistent results across multiple calls."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.Coefficient(model)

    # Call attributions multiple times
    attributions1 = explainer.attributions(X_test)
    attributions2 = explainer.attributions(X_test)

    # Check consistency
    assert np.allclose(attributions1, attributions2)


def test_coefficient_with_different_sample_sizes():
    """Test Coefficient explainer with different sample sizes."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.Coefficient(model)

    # Test with different sample sizes
    for n_samples in [1, 5, 10, 20]:
        attributions = explainer.attributions(X_test[:n_samples])
        assert attributions.shape == (n_samples, X_test.shape[1])

        # All rows should be identical
        for i in range(n_samples):
            assert np.allclose(attributions[i], model.coef_)


def test_coefficient_all_rows_identical():
    """Test that all rows in the output are identical."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.Coefficient(model)
    attributions = explainer.attributions(X_test)

    # Check that all rows are identical
    for i in range(1, len(X_test)):
        assert np.allclose(attributions[i], attributions[0])


def test_coefficient_with_intercept():
    """Test Coefficient explainer with models that have an intercept."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.Coefficient(model)
    attributions = explainer.attributions(X_test)

    # Attributions should only contain coefficients, not intercept
    assert attributions.shape == X_test.shape
    assert np.allclose(attributions[0], model.coef_)


def test_coefficient_without_intercept():
    """Test Coefficient explainer with models without an intercept."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    model = sklearn.linear_model.LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.Coefficient(model)
    attributions = explainer.attributions(X_test)

    assert attributions.shape == X_test.shape
    assert np.allclose(attributions[0], model.coef_)


def test_coefficient_with_normalized_data():
    """Test Coefficient explainer with normalized data."""
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0), test_size=0.2, random_state=0
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train_scaled, y_train)

    explainer = shap.explainers.other.Coefficient(model)
    attributions = explainer.attributions(X_test_scaled)

    assert attributions.shape == X_test_scaled.shape
    assert np.allclose(attributions[0], model.coef_)
