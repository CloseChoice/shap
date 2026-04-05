"""Unit tests for the Permutation explainer."""

import pickle

import numpy as np
import pytest

import shap

from . import common


def test_exact_second_order(random_seed):
    """This tests that the Perumtation explain gives exact answers for second order functions."""
    rs = np.random.RandomState(random_seed)
    data = rs.randint(0, 2, size=(100, 5))

    def model(data):
        return data[:, 0] * data[:, 2] + data[:, 1] + data[:, 2] + data[:, 2] * data[:, 3]

    right_answer = np.zeros(data.shape)
    right_answer[:, 0] += (data[:, 0] * data[:, 2]) / 2
    right_answer[:, 2] += (data[:, 0] * data[:, 2]) / 2
    right_answer[:, 1] += data[:, 1]
    right_answer[:, 2] += data[:, 2]
    right_answer[:, 2] += (data[:, 2] * data[:, 3]) / 2
    right_answer[:, 3] += (data[:, 2] * data[:, 3]) / 2
    shap_values = shap.explainers.PermutationExplainer(model, np.zeros((1, 5)))(data)

    assert np.allclose(right_answer, shap_values.values)  # type: ignore[union-attr]


def test_tabular_single_output_auto_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.PermutationExplainer, model.predict, data, data)


def test_tabular_multi_output_auto_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.PermutationExplainer, model.predict_proba, data, data)


def test_tabular_single_output_partition_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.PermutationExplainer, model.predict, shap.maskers.Partition(data), data)


def test_tabular_multi_output_partition_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(
        shap.explainers.PermutationExplainer, model.predict_proba, shap.maskers.Partition(data), data
    )


def test_tabular_single_output_independent_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.PermutationExplainer, model.predict, shap.maskers.Independent(data), data)


def test_tabular_multi_output_independent_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(
        shap.explainers.PermutationExplainer, model.predict_proba, shap.maskers.Independent(data), data
    )


def test_serialization():
    model, data = common.basic_xgboost_scenario()
    common.test_serialization(
        shap.explainers.PermutationExplainer, model.predict, data, data, rtol=0.1, atol=0.05, max_evals=100000
    )


def test_serialization_no_model_or_masker():
    model, data = common.basic_xgboost_scenario()
    common.test_serialization(
        shap.explainers.PermutationExplainer,
        model.predict,
        data,
        data,
        model_saver=False,
        masker_saver=False,
        model_loader=lambda _: model.predict,
        masker_loader=lambda _: data,
        rtol=0.1,
        atol=0.05,
        max_evals=100000,
    )


def test_serialization_custom_model_save():
    model, data = common.basic_xgboost_scenario()
    common.test_serialization(
        shap.explainers.PermutationExplainer,
        model.predict,
        data,
        data,
        model_saver=pickle.dump,
        model_loader=pickle.load,
        rtol=0.1,
        atol=0.05,
        max_evals=100000,
    )


def test_permutation_explainer_masker_none_error():
    """Test that PermutationExplainer raises error when masker is None."""
    def model(x):
        return x.sum(axis=1)

    with pytest.raises(ValueError, match="masker cannot be None"):
        shap.explainers.PermutationExplainer(model, masker=None)


def test_permutation_explainer_with_seed():
    """Test that PermutationExplainer is reproducible with a seed."""
    def model(x):
        return x.sum(axis=1)

    X_background = np.random.randn(20, 5)
    X_test = np.random.randn(3, 5)

    # Run with seed=42
    explainer1 = shap.explainers.PermutationExplainer(model, X_background, seed=42)
    explanation1 = explainer1(X_test, max_evals=500)

    # Run with same seed
    explainer2 = shap.explainers.PermutationExplainer(model, X_background, seed=42)
    explanation2 = explainer2(X_test, max_evals=500)

    # Results should be identical
    assert np.allclose(explanation1.values, explanation2.values)


def test_permutation_explainer_error_bounds():
    """Test that PermutationExplainer returns error bounds when requested."""
    def model(x):
        return 2 * x[:, 0] + x[:, 1]

    X_background = np.random.randn(20, 3)
    X_test = np.random.randn(2, 3)

    explainer = shap.explainers.PermutationExplainer(model, X_background)
    explanation = explainer(X_test, max_evals=1000, error_bounds=True)

    # Check that error bounds are present
    assert hasattr(explanation, "error_std")
    assert explanation.error_std is not None
    assert explanation.error_std.shape == explanation.values.shape


def test_permutation_explainer_main_effects():
    """Test that PermutationExplainer computes main effects when requested."""
    def model(x):
        return x.sum(axis=1)

    X_background = np.random.randn(20, 4)
    X_test = np.random.randn(2, 4)

    explainer = shap.explainers.PermutationExplainer(model, X_background)
    explanation = explainer(X_test, max_evals=500, main_effects=True)

    # Check that the call succeeds with main_effects=True
    assert explanation.values is not None
    assert explanation.values.shape == X_test.shape


def test_permutation_explainer_shap_values_method():
    """Test the legacy shap_values method."""
    def model(x):
        return x[:, 0] * 2 + x[:, 1]

    X_background = np.random.randn(20, 3)
    X_test = np.random.randn(5, 3)

    explainer = shap.explainers.PermutationExplainer(model, X_background)
    shap_values = explainer.shap_values(X_test, npermutations=10)

    # Check shape
    assert shap_values.shape == X_test.shape

    # Check that it's similar to using __call__
    explanation = explainer(X_test, max_evals=10 * X_test.shape[1])
    assert np.allclose(shap_values, explanation.values, atol=0.1)


def test_permutation_explainer_with_call_args():
    """Test PermutationExplainer with default call arguments."""
    def model(x):
        return x.sum(axis=1)

    X_background = np.random.randn(20, 4)
    X_test = np.random.randn(2, 4)

    # Create explainer with default max_evals
    explainer = shap.explainers.PermutationExplainer(model, X_background, max_evals=200)
    explanation = explainer(X_test)  # Should use max_evals=200 by default

    assert explanation.values is not None
    assert explanation.values.shape == X_test.shape


def test_permutation_explainer_max_evals_too_low():
    """Test that PermutationExplainer raises error when max_evals is too low."""
    def model(x):
        return x.sum(axis=1)

    X_background = np.random.randn(20, 5)
    X_test = np.random.randn(1, 5)

    explainer = shap.explainers.PermutationExplainer(model, X_background)

    # max_evals must be at least 2 * num_features + 1 = 11 for 5 features
    with pytest.raises(ValueError, match="max_evals.*is too low"):
        explainer(X_test, max_evals=5)


def test_permutation_explainer_no_varying_features():
    """Test PermutationExplainer when no features vary from background."""
    def model(x):
        return x.sum(axis=1)

    # Create background and test data that are identical
    X_background = np.ones((20, 3))
    X_test = np.ones((1, 3))  # Same as background

    explainer = shap.explainers.PermutationExplainer(model, X_background)
    explanation = explainer(X_test, max_evals=100)

    # SHAP values should be near zero when features don't vary
    assert np.abs(explanation.values).max() < 0.01


def test_permutation_explainer_max_evals_auto():
    """Test PermutationExplainer with max_evals='auto'."""
    def model(x):
        return x[:, 0] + x[:, 1]

    X_background = np.random.randn(20, 3)
    X_test = np.random.randn(2, 3)

    explainer = shap.explainers.PermutationExplainer(model, X_background)
    explanation = explainer(X_test, max_evals="auto")

    # Should complete without error
    assert explanation.values is not None
    assert explanation.values.shape == X_test.shape


def test_permutation_explainer_str_method():
    """Test the __str__ method of PermutationExplainer."""
    def model(x):
        return x.sum(axis=1)

    X_background = np.random.randn(10, 2)

    explainer = shap.explainers.PermutationExplainer(model, X_background)
    str_repr = str(explainer)

    assert "PermutationExplainer" in str_repr


def test_permutation_explainer_no_varying_features_with_error_bounds():
    """Test PermutationExplainer with error_bounds when no features vary."""
    def model(x):
        return x.sum(axis=1)

    # Create background and test data that are identical
    X_background = np.ones((20, 3))
    X_test = np.ones((1, 3))

    explainer = shap.explainers.PermutationExplainer(model, X_background)
    explanation = explainer(X_test, max_evals=100, error_bounds=True)

    # Should handle the edge case properly
    assert explanation.values is not None
    assert hasattr(explanation, "error_std")


def test_permutation_explainer_with_partition_masker_clustering():
    """Test PermutationExplainer with Partition masker that has clustering."""
    def model(x):
        return 2 * x[:, 0] + x[:, 1] + 0.5 * x[:, 2]

    X_background = np.random.randn(30, 3)
    X_test = np.random.randn(2, 3)

    # Create a Partition masker with clustering
    masker = shap.maskers.Partition(X_background, clustering="correlation")

    explainer = shap.explainers.PermutationExplainer(model, masker)
    explanation = explainer(X_test, max_evals=200)

    # Should complete without error and use clustering
    assert explanation.values is not None
    assert explanation.values.shape == X_test.shape
    assert hasattr(explanation, "clustering")
