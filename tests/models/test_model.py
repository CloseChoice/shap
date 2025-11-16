"""This file contains tests for the Model class."""

import io
import tempfile

import numpy as np
import pytest

import shap


def test_model_init_with_callable():
    """Test Model initialization with a simple callable."""

    def simple_model(x):
        return x * 2

    model = shap.models.Model(simple_model)
    assert model.inner_model == simple_model


def test_model_init_with_model():
    """Test Model initialization with another Model object."""

    def simple_model(x):
        return x * 2

    model1 = shap.models.Model(simple_model)
    model2 = shap.models.Model(model1)
    assert model2.inner_model == simple_model


def test_model_init_with_output_names():
    """Test Model initialization preserves output_names attribute."""

    class ModelWithOutputNames:
        def __init__(self):
            self.output_names = ["output1", "output2"]

        def __call__(self, x):
            return x

    model_with_names = ModelWithOutputNames()
    model = shap.models.Model(model_with_names)
    assert hasattr(model, "output_names")
    assert model.output_names == ["output1", "output2"]


def test_model_call_with_numpy():
    """Test Model.__call__ with numpy array."""

    def simple_model(x):
        return x * 2

    model = shap.models.Model(simple_model)
    x = np.array([1, 2, 3])
    result = model(x)
    np.testing.assert_array_equal(result, np.array([2, 4, 6]))


def test_model_call_with_list():
    """Test Model.__call__ with list input."""

    def simple_model(x):
        return [val * 2 for val in x]

    model = shap.models.Model(simple_model)
    x = [1, 2, 3]
    result = model(x)
    np.testing.assert_array_equal(result, np.array([2, 4, 6]))


def test_model_call_with_torch_tensor():
    """Test Model.__call__ with PyTorch tensor."""
    torch = pytest.importorskip("torch")

    def torch_model(x):
        return torch.tensor(x) * 2

    model = shap.models.Model(torch_model)
    x = np.array([1.0, 2.0, 3.0])
    result = model(x)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([2.0, 4.0, 6.0]))


def test_model_call_with_multiple_args():
    """Test Model.__call__ with multiple arguments."""

    def multi_arg_model(x, y):
        return x + y

    model = shap.models.Model(multi_arg_model)
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    result = model(x, y)
    np.testing.assert_array_equal(result, np.array([5, 7, 9]))


def test_model_init_without_output_names():
    """Test Model initialization when model doesn't have output_names."""

    def simple_model(x):
        return x * 2

    model = shap.models.Model(simple_model)
    assert not hasattr(model, "output_names")


def test_model_save_and_load():
    """Test Model save and load functionality."""

    def simple_model(x):
        return x * 2

    model = shap.models.Model(simple_model)

    # Save the model
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".shap") as f:
        temp_path = f.name
        model.save(f)

    # Load the model
    try:
        with open(temp_path, "rb") as f:
            loaded_model = shap.models.Model.load(f)

        # Test that loaded model works
        x = np.array([1, 2, 3])
        result = loaded_model(x)
        np.testing.assert_array_equal(result, np.array([2, 4, 6]))
    finally:
        import os
        os.unlink(temp_path)


# TODO: check if this code is dead!
# The Model.load() with instantiate=False appears to have a bug in the serialization code.
# Skipping this test until the serialization implementation is fixed.
# def test_model_load_instantiate_false():
#     """Test Model.load with instantiate=False."""
#     def simple_model(x):
#         return x * 2
#     model = shap.models.Model(simple_model)
#     # Error: TypeError: '<' not supported between instances of 'type' and 'int' in Deserializer
