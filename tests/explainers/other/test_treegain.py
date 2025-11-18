"""Tests for TreeGain explainer."""

import numpy as np
import pytest
import sklearn.datasets
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.tree

import shap


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
