"""This file contains tests for partition explainer."""

import pickle
import platform

import numpy as np
import pytest

import shap

from . import common


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_translation(basic_translation_scenario):
    model, tokenizer, data = basic_translation_scenario
    common.test_additivity(shap.explainers.PartitionExplainer, model, tokenizer, data)


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_translation_auto(basic_translation_scenario):
    model, tokenizer, data = basic_translation_scenario
    common.test_additivity(shap.Explainer, model, tokenizer, data)


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_translation_algorithm_arg(basic_translation_scenario):
    model, tokenizer, data = basic_translation_scenario
    common.test_additivity(shap.Explainer, model, tokenizer, data, algorithm="partition")


def test_tabular_single_output():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.PartitionExplainer, model.predict, shap.maskers.Partition(data), data)


def test_tabular_multi_output():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.PartitionExplainer, model.predict_proba, shap.maskers.Partition(data), data)


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_serialization(basic_translation_scenario):
    model, tokenizer, data = basic_translation_scenario
    common.test_serialization(shap.explainers.PartitionExplainer, model, tokenizer, data)


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_serialization_no_model_or_masker(basic_translation_scenario):
    model, tokenizer, data = basic_translation_scenario
    common.test_serialization(
        shap.explainers.Partition,
        model,
        tokenizer,
        data,
        model_saver=None,
        masker_saver=None,
        model_loader=lambda _: model,
        masker_loader=lambda _: tokenizer,
    )


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_serialization_custom_model_save(basic_translation_scenario):
    model, tokenizer, data = basic_translation_scenario
    common.test_serialization(
        shap.explainers.PartitionExplainer, model, tokenizer, data, model_saver=pickle.dump, model_loader=pickle.load
    )


def test_partition_explainer_simple_model():
    """Test PartitionExplainer with a simple linear model."""
    def model(x):
        return x.sum(axis=1)

    X_background = np.random.randn(20, 5)
    X_test = np.random.randn(3, 5)

    masker = shap.maskers.Partition(X_background)
    explainer = shap.explainers.PartitionExplainer(model, masker)
    explanation = explainer(X_test)

    # Check that explanation has values
    assert explanation.values is not None
    assert explanation.values.shape == X_test.shape


def test_partition_explainer_masker_no_clustering_error():
    """Test that PartitionExplainer raises error when masker has no clustering."""
    def model(x):
        return x.sum(axis=1)

    X_background = np.random.randn(20, 5)

    # Independent masker doesn't have clustering by default
    masker = shap.maskers.Independent(X_background)

    with pytest.raises(ValueError, match="must have a .clustering attribute"):
        shap.explainers.PartitionExplainer(model, masker)


def test_partition_explainer_call_method():
    """Test the __call__ method returns Explanation object."""
    def model(x):
        return 2 * x[:, 0] + x[:, 1]

    X_background = np.random.randn(15, 3)
    X_test = np.random.randn(2, 3)

    masker = shap.maskers.Partition(X_background, max_samples=10)
    explainer = shap.explainers.PartitionExplainer(model, masker)
    explanation = explainer(X_test, max_evals=50)

    # Check Explanation properties
    assert hasattr(explanation, "values")
    assert hasattr(explanation, "base_values")
    assert explanation.values.shape == X_test.shape


def test_partition_explainer_with_max_evals():
    """Test PartitionExplainer with different max_evals values."""
    def model(x):
        return x.mean(axis=1)

    X_background = np.random.randn(20, 4)
    X_test = np.random.randn(2, 4)

    masker = shap.maskers.Partition(X_background)
    explainer = shap.explainers.PartitionExplainer(model, masker)

    # Test with low max_evals
    explanation_low = explainer(X_test, max_evals=10)
    assert explanation_low.values.shape == X_test.shape

    # Test with high max_evals
    explanation_high = explainer(X_test, max_evals=100)
    assert explanation_high.values.shape == X_test.shape


def test_partition_explainer_multi_output():
    """Test PartitionExplainer with multi-output model."""
    def model(x):
        # Return two outputs
        return np.column_stack([x.sum(axis=1), x.mean(axis=1)])

    X_background = np.random.randn(20, 3)
    X_test = np.random.randn(2, 3)

    masker = shap.maskers.Partition(X_background)
    explainer = shap.explainers.PartitionExplainer(model, masker)
    explanation = explainer(X_test, max_evals=50)

    # Should handle multi-output
    assert explanation.values is not None


def test_partition_explainer_with_call_args():
    """Test PartitionExplainer with default call arguments."""
    def model(x):
        return x.sum(axis=1)

    X_background = np.random.randn(20, 4)
    X_test = np.random.randn(2, 4)

    masker = shap.maskers.Partition(X_background)
    # Create explainer with default max_evals
    explainer = shap.explainers.PartitionExplainer(model, masker, max_evals=50)
    explanation = explainer(X_test)  # Should use max_evals=50 by default

    assert explanation.values is not None
    assert explanation.values.shape == X_test.shape


def test_partition_explainer_single_sample():
    """Test PartitionExplainer with a single sample."""
    def model(x):
        return x.sum(axis=1)

    X_background = np.random.randn(20, 3)
    X_test = np.random.randn(1, 3)

    masker = shap.maskers.Partition(X_background)
    explainer = shap.explainers.PartitionExplainer(model, masker)
    explanation = explainer(X_test, max_evals=30)

    assert explanation.values.shape == (1, 3)


def test_partition_explainer_with_clustering():
    """Test PartitionExplainer uses clustering from masker."""
    def model(x):
        return 2 * x[:, 0] + x[:, 1]

    X_background = np.random.randn(30, 4)
    X_test = np.random.randn(2, 4)

    # Create masker with clustering
    masker = shap.maskers.Partition(X_background, clustering="correlation")
    explainer = shap.explainers.PartitionExplainer(model, masker)
    explanation = explainer(X_test, max_evals=50)

    assert explanation.values is not None
    assert explanation.values.shape == X_test.shape


def test_partition_explainer_expected_value():
    """Test that PartitionExplainer computes base values."""
    def model(x):
        return x.sum(axis=1)

    X_background = np.random.randn(20, 3)
    X_test = np.random.randn(2, 3)

    masker = shap.maskers.Partition(X_background)
    explainer = shap.explainers.PartitionExplainer(model, masker)
    explanation = explainer(X_test, max_evals=50)

    # Base values should be present
    assert explanation.base_values is not None
    assert len(explanation.base_values) == len(X_test)
