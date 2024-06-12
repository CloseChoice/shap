"""Unit tests for the Exact explainer."""

import pickle

import shap

from . import common
import sklearn
from shap import datasets
import pytest


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
        shap.explainers.ExactExplainer, model.predict, data, data,
        model_saver=False, masker_saver=False,
        model_loader=lambda _: model.predict, masker_loader=lambda _: data
    )

def test_serialization_custom_model_save():
    model, data = common.basic_xgboost_scenario()
    common.test_serialization(
        shap.explainers.ExactExplainer, model.predict, data, data,
        model_saver=pickle.dump, model_loader=pickle.load
    )

def test_exact_explainer():
    X, y = shap.datasets.iris(n_points=100)

    rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
    rfc.fit(X, y)


    explainer = shap.ExactExplainer(rfc.predict, X.iloc[:2])
    #          sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
    # 114                5.8               2.8                5.1               2.4

    #      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
    # 114                5.8               2.8                5.1               2.4
    # 62                 6.0               2.2                4.0               1.0
    shap_values = explainer(X.iloc[:1])
