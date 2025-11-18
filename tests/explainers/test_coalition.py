"""This file contains tests for coalition explainer."""

import numpy as np
import pandas as pd
import pytest

import shap
from shap.explainers._coalition import create_partition_hierarchy

from . import common


def test_tabular_coalition_single_output():
    coalition_tree = {
        "Demographics": ["Sex", "Age", "Race", "Marital Status", "Education-Num"],
        "Work": ["Occupation", "Workclass", "Hours per week"],
        "Finance": ["Capital Gain", "Capital Loss"],
        "Residence": ["Country"],
    }
    model, data = common.basic_xgboost_scenario(100)
    X, _ = shap.datasets.adult()
    features = X.columns.tolist()
    masker = shap.maskers.Partition(data)
    masker.feature_names = features
    common.test_additivity(
        shap.explainers.CoalitionExplainer, model.predict, masker, data, partition_tree=coalition_tree
    )


def test_tabular_coalition_multiple_output():
    coalition_tree = {
        "Demographics": ["Sex", "Age", "Race", "Marital Status", "Education-Num"],
        "Work": ["Occupation", "Workclass", "Hours per week"],
        "Finance": ["Capital Gain", "Capital Loss"],
        "Residence": ["Country"],
    }
    model, data = common.basic_xgboost_scenario(100)
    X, _ = shap.datasets.adult()
    features = X.columns.tolist()
    masker = shap.maskers.Partition(data)
    masker.feature_names = features
    common.test_additivity(
        shap.explainers.CoalitionExplainer, model.predict_proba, masker, data, partition_tree=coalition_tree
    )


def test_tabular_coalition_exact_match():
    model, data = common.basic_xgboost_scenario(50)
    X, _ = shap.datasets.adult()
    features = X.columns.tolist()
    data = pd.DataFrame(data, columns=features)
    exact_explainer = shap.explainers.ExactExplainer(model.predict, data)
    shap_values = exact_explainer(data)

    flat_hierarchy = {}
    for name in features:
        flat_hierarchy[name] = name

    partition_masker = shap.maskers.Partition(data)
    partition_masker.feature_names = features
    partition_explainer_f = shap.CoalitionExplainer(model.predict, partition_masker, partition_tree=flat_hierarchy)
    flat_winter_values = partition_explainer_f(data)
    assert np.allclose(shap_values.values, flat_winter_values.values)


def test_tabular_coalition_partition_match():
    model, data = common.basic_xgboost_scenario(50)
    X, _ = shap.datasets.adult()
    features = X.columns.tolist()
    data = pd.DataFrame(data, columns=features)
    partition_tree = shap.utils.partition_tree(data)
    partition_masker = shap.maskers.Partition(data, clustering=partition_tree)
    partition_masker.feature_names = features
    partition_explainer = shap.explainers.PartitionExplainer(model.predict, partition_masker)
    binary_values = partition_explainer(data)

    hierarchy_binary = create_partition_hierarchy(partition_tree, features)

    coalition_masker = shap.maskers.Partition(data)
    partition_explainer_b = shap.CoalitionExplainer(model.predict, coalition_masker, partition_tree=hierarchy_binary)  # type: ignore[arg-type]
    binary_winter_values = partition_explainer_b(data)

    assert np.allclose(binary_values.values, binary_winter_values.values)  # type: ignore[union-attr]


def test_coalition_explainer_simple():
    """Test CoalitionExplainer with a simple model and coalition tree."""
    def model(x):
        return x.sum(axis=1)

    # Create synthetic data
    X = np.random.randn(20, 4)
    feature_names = ["f1", "f2", "f3", "f4"]

    # Define a simple coalition tree
    coalition_tree = {
        "Group1": ["f1", "f2"],
        "Group2": ["f3", "f4"]
    }

    # Create masker with feature names
    masker = shap.maskers.Partition(X)
    masker.feature_names = feature_names

    # Create explainer
    explainer = shap.explainers.CoalitionExplainer(model, masker, partition_tree=coalition_tree)

    # Get explanations
    X_test = np.random.randn(2, 4)
    explanation = explainer(X_test)

    # Check that explanation has values
    assert explanation.values is not None
    assert explanation.values.shape == X_test.shape


def test_coalition_explainer_multi_output():
    """Test CoalitionExplainer with multi-output model."""
    def model(x):
        return np.column_stack([x.sum(axis=1), x.mean(axis=1)])

    X = np.random.randn(15, 4)
    feature_names = ["a", "b", "c", "d"]

    coalition_tree = {
        "AB": ["a", "b"],
        "CD": ["c", "d"]
    }

    masker = shap.maskers.Partition(X)
    masker.feature_names = feature_names

    explainer = shap.explainers.CoalitionExplainer(model, masker, partition_tree=coalition_tree)

    X_test = np.random.randn(2, 4)
    explanation = explainer(X_test)

    # Should handle multi-output
    assert explanation.values is not None


def test_coalition_explainer_single_sample():
    """Test CoalitionExplainer with single sample."""
    def model(x):
        return x.sum(axis=1)

    X = np.random.randn(15, 3)
    feature_names = ["x", "y", "z"]

    coalition_tree = {
        "XY": ["x", "y"],
        "Z": ["z"]
    }

    masker = shap.maskers.Partition(X)
    masker.feature_names = feature_names

    explainer = shap.explainers.CoalitionExplainer(model, masker, partition_tree=coalition_tree)

    X_test = np.random.randn(1, 3)
    explanation = explainer(X_test)

    assert explanation.values.shape == (1, 3)


def test_coalition_explainer_three_level_hierarchy():
    """Test CoalitionExplainer with three-level hierarchy."""
    def model(x):
        return 2 * x[:, 0] + x[:, 1] + 0.5 * x[:, 2] + x[:, 3]

    X = np.random.randn(20, 4)
    feature_names = ["f1", "f2", "f3", "f4"]

    # Nested coalition tree
    coalition_tree = {
        "All": {
            "Group1": ["f1", "f2"],
            "Group2": ["f3", "f4"]
        }
    }

    masker = shap.maskers.Partition(X)
    masker.feature_names = feature_names

    explainer = shap.explainers.CoalitionExplainer(model, masker, partition_tree=coalition_tree)

    X_test = np.random.randn(2, 4)
    explanation = explainer(X_test)

    assert explanation.values is not None
    assert explanation.values.shape == X_test.shape


def test_coalition_explainer_with_dataframe():
    """Test CoalitionExplainer with DataFrame input."""
    def model(x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        return x.sum(axis=1)

    # Create DataFrame
    X = pd.DataFrame(np.random.randn(20, 4), columns=["a", "b", "c", "d"])

    coalition_tree = {
        "AB": ["a", "b"],
        "CD": ["c", "d"]
    }

    masker = shap.maskers.Partition(X)

    explainer = shap.explainers.CoalitionExplainer(model, masker, partition_tree=coalition_tree)

    X_test = pd.DataFrame(np.random.randn(2, 4), columns=["a", "b", "c", "d"])
    explanation = explainer(X_test)

    assert explanation.values is not None
    assert explanation.values.shape == X_test.shape
