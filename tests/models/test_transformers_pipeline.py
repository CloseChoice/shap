"""This file contains tests for the TransformersPipeline class."""

import platform

import numpy as np
import pytest

import shap


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_transformers_pipeline_init():
    """Test TransformersPipeline initialization."""
    transformers = pytest.importorskip("transformers")

    # Use a tiny model for testing
    name = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
    pipeline = transformers.pipeline("text-classification", model=name, top_k=None)

    wrapped_pipeline = shap.models.TransformersPipeline(pipeline)
    assert wrapped_pipeline.inner_model == pipeline
    assert hasattr(wrapped_pipeline, "label2id")
    assert hasattr(wrapped_pipeline, "id2label")
    assert hasattr(wrapped_pipeline, "output_shape")
    assert hasattr(wrapped_pipeline, "output_names")


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_transformers_pipeline_init_with_rescale():
    """Test TransformersPipeline initialization with rescale_to_logits."""
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
    pipeline = transformers.pipeline("text-classification", model=name, top_k=None)

    wrapped_pipeline = shap.models.TransformersPipeline(pipeline, rescale_to_logits=True)
    assert wrapped_pipeline.rescale_to_logits is True


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_transformers_pipeline_call():
    """Test TransformersPipeline.__call__ method."""
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
    pipeline = transformers.pipeline("text-classification", model=name, top_k=None)

    wrapped_pipeline = shap.models.TransformersPipeline(pipeline)

    # Test with list of strings
    texts = ["This is a test", "Another test"]
    output = wrapped_pipeline(texts)

    assert isinstance(output, np.ndarray)
    assert output.shape[0] == 2  # Number of inputs
    assert len(output.shape) == 2  # Should be 2D array
    # Check that probabilities sum to approximately 1 for each input
    assert np.allclose(output.sum(axis=1), 1.0, atol=0.1)


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_transformers_pipeline_call_with_rescale():
    """Test TransformersPipeline.__call__ with rescale_to_logits."""
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
    pipeline = transformers.pipeline("text-classification", model=name, top_k=None)

    wrapped_pipeline = shap.models.TransformersPipeline(pipeline, rescale_to_logits=True)

    texts = ["This is a test"]
    output = wrapped_pipeline(texts)

    assert isinstance(output, np.ndarray)
    # Logits should not sum to 1 (unlike probabilities)
    assert not np.allclose(output.sum(axis=1), 1.0, atol=0.1)


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_transformers_pipeline_call_single_string_error():
    """Test that TransformersPipeline raises error for single string input."""
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
    pipeline = transformers.pipeline("text-classification", model=name, top_k=None)

    wrapped_pipeline = shap.models.TransformersPipeline(pipeline)

    # Should raise assertion error for single string
    with pytest.raises(AssertionError, match="expects a list of strings"):
        wrapped_pipeline("This is a test")


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_transformers_pipeline_label2id_conversion():
    """Test that TransformersPipeline correctly converts label2id to int."""
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
    pipeline = transformers.pipeline("text-classification", model=name, top_k=None)

    wrapped_pipeline = shap.models.TransformersPipeline(pipeline)

    # Verify all values in label2id are integers
    for v in wrapped_pipeline.label2id.values():
        assert isinstance(v, int)


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_transformers_pipeline_output_shape():
    """Test that TransformersPipeline correctly determines output_shape."""
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
    pipeline = transformers.pipeline("text-classification", model=name, top_k=None)

    wrapped_pipeline = shap.models.TransformersPipeline(pipeline)

    # Output shape should match the max label id + 1
    expected_shape = (max(wrapped_pipeline.label2id.values()) + 1,)
    assert wrapped_pipeline.output_shape == expected_shape


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_transformers_pipeline_get_with_unknown_label():
    """Test that TransformersPipeline handles unknown labels in id2label."""
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
    pipeline = transformers.pipeline("text-classification", model=name, top_k=None)

    wrapped_pipeline = shap.models.TransformersPipeline(pipeline)

    # Test that output_names uses .get() with "Unknown" default
    # This tests line 26: self.id2label.get(i, "Unknown")
    # If there are gaps in id2label, they should be "Unknown"
    for i, name in enumerate(wrapped_pipeline.output_names):
        expected = wrapped_pipeline.id2label.get(i, "Unknown")
        assert name == expected


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_transformers_pipeline_with_single_result():
    """Test TransformersPipeline handles single result (not a list) from pipeline."""
    transformers = pytest.importorskip("transformers")

    # Create pipeline without top_k to get single result per input
    name = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
    pipeline = transformers.pipeline("text-classification", model=name)

    wrapped_pipeline = shap.models.TransformersPipeline(pipeline)

    texts = ["This is a test"]
    output = wrapped_pipeline(texts)

    assert isinstance(output, np.ndarray)
    assert output.shape[0] == 1  # One input


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_transformers_pipeline_output_names():
    """Test that TransformersPipeline correctly sets output_names."""
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
    pipeline = transformers.pipeline("text-classification", model=name, top_k=None)

    wrapped_pipeline = shap.models.TransformersPipeline(pipeline)

    assert hasattr(wrapped_pipeline, "output_names")
    assert isinstance(wrapped_pipeline.output_names, list)
    assert len(wrapped_pipeline.output_names) == wrapped_pipeline.output_shape[0]
