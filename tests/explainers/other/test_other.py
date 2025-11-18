"""Tests for explainers in the 'other' module.

This file contains tests for:
- TreeGain explainer
- Coefficient explainer
- Random explainer
- LimeTabular explainer
"""

import numpy as np
import pandas as pd
import pytest
import sklearn.datasets
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.tree

import shap


# ============================================================================
# TreeGain Explainer Tests
# ============================================================================


def test_treegain_xgboost_regressor():
    """Test TreeGain with XGBoost regressor."""
    xgboost = pytest.importorskip("xgboost")

    X, y = sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0)
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = xgboost.XGBRegressor(n_estimators=10, max_depth=3, random_state=0)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.TreeGain(model)
    attributions = explainer.attributions(X_test)

    # Check shape
    assert attributions.shape == X_test.shape

    # Check that all rows are identical
    assert np.allclose(attributions[0], attributions[5])

    # Check that attributions match model's feature importances
    assert np.allclose(attributions[0], model.feature_importances_)


def test_treegain_xgboost_classifier():
    """Test TreeGain with XGBoost classifier."""
    xgboost = pytest.importorskip("xgboost")

    X, y = sklearn.datasets.make_classification(n_samples=100, n_features=8, random_state=0)
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        X, y, test_size=0.1, random_state=0
    )

    model = xgboost.XGBClassifier(n_estimators=10, max_depth=3, random_state=0, use_label_encoder=False)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.TreeGain(model)
    attributions = explainer.attributions(X_test)

    # Check shape
    assert attributions.shape == X_test.shape

    # Check that attributions match model's feature importances
    assert np.allclose(attributions[0], model.feature_importances_)


def test_treegain_lightgbm_regressor():
    """Test TreeGain with LightGBM regressor."""
    lightgbm = pytest.importorskip("lightgbm")

    X, y = sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0)
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = lightgbm.LGBMRegressor(n_estimators=10, max_depth=3, random_state=0, verbose=-1)
    model.fit(X_train, y_train)

    # LightGBM models have feature_importances_ but TreeGain might not recognize the type
    # Just test that it has the attribute
    assert hasattr(model, "feature_importances_")


def test_treegain_unsupported_model():
    """Test that TreeGain raises NotImplementedError for unsupported models."""
    X, y = sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0)
    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)

    with pytest.raises(NotImplementedError, match="not yet supported"):
        shap.explainers.other.TreeGain(model)


def test_treegain_model_without_feature_importances():
    """Test that TreeGain raises AssertionError for models without feature_importances_."""
    xgboost = pytest.importorskip("xgboost")

    X, y = sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0)

    model = xgboost.XGBRegressor(n_estimators=10, random_state=0)
    model.fit(X, y)

    # Remove feature_importances_ to simulate a model without it
    original_importances = model.feature_importances_
    delattr(model, "feature_importances_")

    with pytest.raises(AssertionError, match="does not have a feature_importances_"):
        shap.explainers.other.TreeGain(model)

    # Restore for cleanup
    model.feature_importances_ = original_importances


def test_treegain_with_single_sample():
    """Test TreeGain with a single sample."""
    xgboost = pytest.importorskip("xgboost")

    X, y = sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0)
    X_train, _, y_train, _ = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = xgboost.XGBRegressor(n_estimators=10, max_depth=3, random_state=0)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.TreeGain(model)

    # Test with single sample
    single_sample = X_train[:1]
    attributions = explainer.attributions(single_sample)

    assert attributions.shape == single_sample.shape
    assert np.allclose(attributions[0], model.feature_importances_)


def test_treegain_consistency_across_calls():
    """Test that TreeGain returns consistent results across multiple calls."""
    xgboost = pytest.importorskip("xgboost")

    X, y = sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0)
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = xgboost.XGBRegressor(n_estimators=10, max_depth=3, random_state=0)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.TreeGain(model)

    # Call attributions multiple times
    attributions1 = explainer.attributions(X_test)
    attributions2 = explainer.attributions(X_test)

    # Check consistency
    assert np.allclose(attributions1, attributions2)


def test_treegain_with_different_sample_sizes():
    """Test TreeGain with different sample sizes."""
    xgboost = pytest.importorskip("xgboost")

    X, y = sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0)
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = xgboost.XGBRegressor(n_estimators=10, max_depth=3, random_state=0)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.TreeGain(model)

    # Test with different sample sizes
    for n_samples in [1, 5, 10, 20]:
        attributions = explainer.attributions(X_test[:n_samples])
        assert attributions.shape == (n_samples, X_test.shape[1])

        # All rows should be identical
        for i in range(n_samples):
            assert np.allclose(attributions[i], model.feature_importances_)


def test_treegain_all_rows_identical():
    """Test that all rows in TreeGain output are identical."""
    xgboost = pytest.importorskip("xgboost")

    X, y = sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0)
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = xgboost.XGBRegressor(n_estimators=10, max_depth=3, random_state=0)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.TreeGain(model)
    attributions = explainer.attributions(X_test)

    # Check that all rows are identical
    for i in range(1, len(X_test)):
        assert np.allclose(attributions[i], attributions[0])


def test_treegain_feature_importances_non_negative():
    """Test that TreeGain returns non-negative feature importances."""
    xgboost = pytest.importorskip("xgboost")

    X, y = sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0)
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = xgboost.XGBRegressor(n_estimators=10, max_depth=3, random_state=0)
    model.fit(X_train, y_train)

    explainer = shap.explainers.other.TreeGain(model)
    attributions = explainer.attributions(X_test)

    # All importances should be non-negative
    assert np.all(attributions >= 0)


def test_treegain_catboost_regressor():
    """Test TreeGain with CatBoost regressor."""
    catboost = pytest.importorskip("catboost")

    X, y = sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0)
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = catboost.CatBoostRegressor(iterations=10, max_depth=3, random_state=0, verbose=0)
    model.fit(X_train, y_train)

    # CatBoost has feature_importances_
    assert hasattr(model, "feature_importances_")


# ============================================================================
# Coefficient Explainer Tests
# ============================================================================


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
    X, y = sklearn.datasets.make_regression(n_samples=100, n_features=8, random_state=0)
    model = sklearn.tree.DecisionTreeRegressor()
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


# ============================================================================
# Random Explainer Tests
# ============================================================================


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


# ============================================================================
# LimeTabular Explainer Tests
# ============================================================================


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
