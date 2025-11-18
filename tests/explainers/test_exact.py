"""Unit tests for the Exact explainer."""

import pickle

import numpy as np
import pytest

import shap

from . import common


def test_interactions():
    model, data = common.basic_xgboost_scenario(100)
    common.test_interactions_additivity(shap.explainers.ExactExplainer, model.predict, data, data)


def test_tabular_single_output_auto_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.ExactExplainer, model.predict, data, data)


def test_tabular_multi_output_auto_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.ExactExplainer, model.predict_proba, data, data)


def test_tabular_single_output_partition_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.ExactExplainer, model.predict, shap.maskers.Partition(data), data)


def test_tabular_multi_output_partition_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.ExactExplainer, model.predict_proba, shap.maskers.Partition(data), data)


def test_tabular_single_output_independent_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.ExactExplainer, model.predict, shap.maskers.Independent(data), data)


def test_tabular_multi_output_independent_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.ExactExplainer, model.predict_proba, shap.maskers.Independent(data), data)


def test_serialization():
    model, data = common.basic_xgboost_scenario()
    common.test_serialization(shap.explainers.ExactExplainer, model.predict, data, data)


def test_serialization_no_model_or_masker():
    model, data = common.basic_xgboost_scenario()
    common.test_serialization(
        shap.explainers.ExactExplainer,
        model.predict,
        data,
        data,
        model_saver=False,
        masker_saver=False,
        model_loader=lambda _: model.predict,
        masker_loader=lambda _: data,
    )


def test_serialization_custom_model_save():
    model, data = common.basic_xgboost_scenario()
    common.test_serialization(
        shap.explainers.ExactExplainer, model.predict, data, data, model_saver=pickle.dump, model_loader=pickle.load
    )


def test_exact_explainer_simple_model():
    """Test ExactExplainer with a simple linear model."""
    def model(x):
        return x.sum(axis=1)

    X_background = np.random.randn(10, 5)
    X_test = np.random.randn(2, 5)

    explainer = shap.explainers.ExactExplainer(model, X_background)
    explanation = explainer(X_test)

    # Check that explanation has values
    assert explanation.values is not None
    assert explanation.values.shape == X_test.shape


def test_exact_explainer_with_partition_masker():
    """Test ExactExplainer with Partition masker."""
    def model(x):
        return 2 * x[:, 0] + x[:, 1]

    X_background = np.random.randn(15, 4)
    X_test = np.random.randn(2, 4)

    masker = shap.maskers.Partition(X_background)
    explainer = shap.explainers.ExactExplainer(model, masker)
    explanation = explainer(X_test)

    # Should handle partition masker with clustering
    assert explanation.values is not None
    assert explanation.values.shape == X_test.shape


def test_exact_explainer_call_method():
    """Test the __call__ method returns Explanation object."""
    def model(x):
        return x[:, 0] * 2 + x[:, 1]

    X_background = np.random.randn(10, 3)
    X_test = np.random.randn(2, 3)

    explainer = shap.explainers.ExactExplainer(model, X_background)
    explanation = explainer(X_test, max_evals=10000)

    # Check Explanation properties
    assert hasattr(explanation, "values")
    assert hasattr(explanation, "base_values")
    assert explanation.values.shape == X_test.shape


def test_exact_explainer_multi_output():
    """Test ExactExplainer with multi-output model."""
    def model(x):
        return np.column_stack([x.sum(axis=1), x.mean(axis=1)])

    X_background = np.random.randn(10, 3)
    X_test = np.random.randn(2, 3)

    explainer = shap.explainers.ExactExplainer(model, X_background)
    explanation = explainer(X_test)

    # Should handle multi-output
    assert explanation.values is not None


def test_exact_explainer_single_sample():
    """Test ExactExplainer with a single sample."""
    def model(x):
        return x.sum(axis=1)

    X_background = np.random.randn(10, 4)
    X_test = np.random.randn(1, 4)

    explainer = shap.explainers.ExactExplainer(model, X_background)
    explanation = explainer(X_test)

    assert explanation.values.shape == (1, 4)


def test_exact_explainer_with_independent_masker():
    """Test ExactExplainer with Independent masker."""
    def model(x):
        return x.sum(axis=1)

    X_background = np.random.randn(15, 4)
    X_test = np.random.randn(2, 4)

    masker = shap.maskers.Independent(X_background)
    explainer = shap.explainers.ExactExplainer(model, masker)
    explanation = explainer(X_test)

    assert explanation.values is not None
    assert explanation.values.shape == X_test.shape


def test_exact_explainer_with_interactions():
    """Test ExactExplainer with interactions parameter."""
    def model(x):
        return x[:, 0] * x[:, 1] + x[:, 2]

    X_background = np.random.randn(10, 3)
    X_test = np.random.randn(2, 3)

    explainer = shap.explainers.ExactExplainer(model, X_background)
    # Request interactions (second-order effects)
    explanation = explainer(X_test, interactions=2)

    # Should compute with interactions
    assert explanation.values is not None


def test_exact_explainer_main_effects():
    """Test ExactExplainer with main_effects parameter."""
    def model(x):
        return x.sum(axis=1)

    X_background = np.random.randn(10, 4)
    X_test = np.random.randn(2, 4)

    explainer = shap.explainers.ExactExplainer(model, X_background)
    explanation = explainer(X_test, main_effects=True)

    # Should complete with main_effects
    assert explanation.values is not None
    assert explanation.values.shape == X_test.shape


def test_exact_explainer_error_bounds():
    """Test ExactExplainer with error_bounds parameter."""
    def model(x):
        return x.mean(axis=1)

    X_background = np.random.randn(10, 3)
    X_test = np.random.randn(2, 3)

    explainer = shap.explainers.ExactExplainer(model, X_background)
    explanation = explainer(X_test, error_bounds=True)

    # Should handle error_bounds parameter
    assert explanation.values is not None
    assert explanation.values.shape == X_test.shape


def test_exact_explainer_small_features():
    """Test ExactExplainer with very few features."""
    def model(x):
        return 3 * x[:, 0] + 2 * x[:, 1]

    X_background = np.random.randn(10, 2)
    X_test = np.random.randn(3, 2)

    explainer = shap.explainers.ExactExplainer(model, X_background)
    explanation = explainer(X_test)

    # Should work well with small feature count
    assert explanation.values.shape == (3, 2)


def test_exact_explainer_additivity():
    """Test that ExactExplainer satisfies additivity property."""
    def model(x):
        return 2 * x[:, 0] + x[:, 1] - 0.5 * x[:, 2]

    X_background = np.random.randn(10, 3)
    X_test = np.random.randn(2, 3)

    explainer = shap.explainers.ExactExplainer(model, X_background)
    explanation = explainer(X_test)

    # SHAP values + base_value should equal model prediction
    predictions = model(X_test)
    reconstructed = explanation.values.sum(axis=1) + explanation.base_values

    assert np.allclose(reconstructed, predictions, atol=1e-5)
