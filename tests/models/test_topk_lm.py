"""This file contains tests for the TopKLM class."""

import io
import platform
import tempfile

import numpy as np
import pytest

import shap


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_init_pytorch():
    """Test TopKLM initialization with PyTorch model."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)

    topk_lm = shap.models.TopKLM(model, tokenizer, k=5)
    assert topk_lm.k == 5
    assert topk_lm.tokenizer == tokenizer
    assert topk_lm.model_type == "pt"
    assert topk_lm.batch_size == 128
    assert topk_lm.X is None
    assert topk_lm.topk_token_ids is None


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_init_with_custom_params():
    """Test TopKLM initialization with custom parameters."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)

    def custom_topk_fn(X):
        return ["token1", "token2", "token3"]

    topk_lm = shap.models.TopKLM(
        model, tokenizer, k=10, generate_topk_token_ids=custom_topk_fn, batch_size=64, device="cpu"
    )
    assert topk_lm.k == 10
    assert topk_lm._custom_generate_topk_token_ids == custom_topk_fn
    assert topk_lm.batch_size == 64
    assert str(topk_lm.device) == "cpu"


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_init_sets_pad_token():
    """Test that TopKLM sets pad_token if not defined."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)

    # Clear pad_token to test setting
    tokenizer.pad_token = None
    topk_lm = shap.models.TopKLM(model, tokenizer, k=5)
    assert topk_lm.tokenizer.pad_token == topk_lm.tokenizer.eos_token


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_update_cache_x():
    """Test TopKLM.update_cache_X method."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)

    topk_lm = shap.models.TopKLM(model, tokenizer, k=3, device="cpu")

    X = np.array(["This is a test"])
    topk_lm.update_cache_X(X)

    assert topk_lm.X is not None
    assert np.array_equal(topk_lm.X, X)
    assert topk_lm.output_names is not None
    assert len(topk_lm.output_names) == 3


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_update_cache_x_multiple_calls():
    """Test that update_cache_X only updates when X changes."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)

    topk_lm = shap.models.TopKLM(model, tokenizer, k=3, device="cpu")

    X1 = np.array(["First test"])
    topk_lm.update_cache_X(X1)
    output_names_1 = topk_lm.output_names

    # Same X should not update
    topk_lm.update_cache_X(X1)
    assert topk_lm.output_names == output_names_1

    # Different X should update
    X2 = np.array(["Second test"])
    topk_lm.update_cache_X(X2)
    assert not np.array_equal(topk_lm.X, X1)


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_generate_topk_token_ids():
    """Test TopKLM.generate_topk_token_ids method."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)

    topk_lm = shap.models.TopKLM(model, tokenizer, k=5, device="cpu")

    X = np.array(["Test input"])
    token_ids = topk_lm.generate_topk_token_ids(X)

    assert isinstance(token_ids, np.ndarray)
    assert len(token_ids) == 5


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_get_inputs():
    """Test TopKLM.get_inputs method."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)

    topk_lm = shap.models.TopKLM(model, tokenizer, k=5, device="cpu")

    X = np.array(["Test input", "Another input"])
    inputs = topk_lm.get_inputs(X)

    assert "input_ids" in inputs
    assert "attention_mask" in inputs


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_get_inputs_with_padding_side():
    """Test TopKLM.get_inputs with custom padding_side."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)

    topk_lm = shap.models.TopKLM(model, tokenizer, k=5, device="cpu")

    X = np.array(["Test"])
    inputs = topk_lm.get_inputs(X, padding_side="left")

    assert "input_ids" in inputs
    # Check padding side is reset to default after call
    assert topk_lm.tokenizer.padding_side == "right"


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_get_lm_logits():
    """Test TopKLM.get_lm_logits method."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)

    topk_lm = shap.models.TopKLM(model, tokenizer, k=5, device="cpu")

    X = np.array(["Test input"])
    logits = topk_lm.get_lm_logits(X)

    assert isinstance(logits, np.ndarray)
    assert logits.dtype == np.float64
    assert logits.shape[0] == 1  # batch size


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_get_logodds():
    """Test TopKLM.get_logodds method."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)

    topk_lm = shap.models.TopKLM(model, tokenizer, k=5, device="cpu")

    # Generate topk_token_ids first
    X = np.array(["Test input"])
    topk_lm.topk_token_ids = topk_lm.generate_topk_token_ids(X)

    # Get logits
    logits = topk_lm.get_lm_logits(X)

    # Test get_logodds
    logodds = topk_lm.get_logodds(logits)

    assert isinstance(logodds, np.ndarray)
    assert logodds.shape[-1] == 5  # k=5


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_call():
    """Test TopKLM.__call__ method."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)

    topk_lm = shap.models.TopKLM(model, tokenizer, k=3, device="cpu")

    X = np.array(["Test input"])
    masked_X = np.array(["Test input", "Another input"])

    output = topk_lm(masked_X, X)

    assert isinstance(output, np.ndarray)
    assert output.shape[0] == 2  # Number of masked inputs
    assert output.shape[-1] == 3  # k=3


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_call_with_batching():
    """Test TopKLM.__call__ with batching."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)

    topk_lm = shap.models.TopKLM(model, tokenizer, k=3, batch_size=2, device="cpu")

    X = np.array(["Test input"])
    masked_X = np.array(["Input 1", "Input 2", "Input 3", "Input 4", "Input 5"])

    output = topk_lm(masked_X, X)

    assert isinstance(output, np.ndarray)
    assert output.shape[0] == 5


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_with_custom_generate_function():
    """Test TopKLM with custom generate_topk_token_ids function."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)

    def custom_topk_fn(X):
        return ["token1", "token2"]

    topk_lm = shap.models.TopKLM(model, tokenizer, k=5, generate_topk_token_ids=custom_topk_fn, device="cpu")

    X = np.array(["Test input"])
    topk_lm.update_cache_X(X)

    # Should use custom function
    assert topk_lm.output_names == ["token1", "token2"]


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_get_output_names():
    """Test TopKLM.get_output_names_and_update_topk_token_ids method."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)

    topk_lm = shap.models.TopKLM(model, tokenizer, k=3, device="cpu")

    X = np.array(["Test input"])
    output_names = topk_lm.get_output_names_and_update_topk_token_ids(X)

    assert isinstance(output_names, list)
    assert len(output_names) == 3
    assert topk_lm.topk_token_ids is not None


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_save_and_load():
    """Test TopKLM save and load functionality."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)

    topk_lm = shap.models.TopKLM(model, tokenizer, k=5, batch_size=64, device="cpu")

    # Save the model
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".shap") as f:
        temp_path = f.name
        topk_lm.save(f)

    # Load the model
    try:
        with open(temp_path, "rb") as f:
            loaded_model = shap.models.TopKLM.load(f)

        assert loaded_model.k == 5
        assert loaded_model.batch_size == 64
    finally:
        import os
        os.unlink(temp_path)


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_get_lm_logits_unsupported_model_type():
    """Test TopKLM.get_lm_logits raises error for unsupported model type."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)

    topk_lm = shap.models.TopKLM(model, tokenizer, k=3, device="cpu")

    # Manually set model_type to invalid value to trigger error
    topk_lm.model_type = "invalid"

    X = np.array(["Test input"])
    with pytest.raises(NotImplementedError, match="Only PyTorch and TensorFlow models are supported"):
        topk_lm.get_lm_logits(X)


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_get_lm_logits_unsupported_model_class():
    """Test TopKLM.get_lm_logits raises error for non-CausalLM model."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    # Use a non-CausalLM model (e.g., classification model)
    name = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(name)

    topk_lm = shap.models.TopKLM(model, tokenizer, k=3, device="cpu")

    X = np.array(["Test input"])
    with pytest.raises(NotImplementedError, match="Model type .* not supported"):
        topk_lm.get_lm_logits(X)


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_update_cache_x_no_change():
    """Test that update_cache_X doesn't regenerate when X is unchanged."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)

    topk_lm = shap.models.TopKLM(model, tokenizer, k=3, device="cpu")

    X = np.array(["Test input"])

    # First update
    topk_lm.update_cache_X(X)
    first_topk_ids = topk_lm.topk_token_ids.copy()
    first_output_names = topk_lm.output_names.copy()

    # Call again with same X - should not regenerate
    topk_lm.update_cache_X(X)

    assert np.array_equal(topk_lm.topk_token_ids, first_topk_ids)
    assert topk_lm.output_names == first_output_names


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_tokenizer_without_pad_token():
    """Test TopKLM handles tokenizer without pad_token."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)

    # Ensure pad_token is None to test the initialization path
    original_pad_token = tokenizer.pad_token
    tokenizer.pad_token = None

    topk_lm = shap.models.TopKLM(model := transformers.AutoModelForCausalLM.from_pretrained(name),
                                  tokenizer, k=3, device="cpu")

    # Should have set pad_token to eos_token
    assert topk_lm.tokenizer.pad_token == topk_lm.tokenizer.eos_token


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
def test_topk_lm_load_instantiate_false():
    """Test TopKLM.load with instantiate=False."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)

    topk_lm = shap.models.TopKLM(model, tokenizer, k=5, batch_size=64, device="cpu")

    # Save the model
    buffer = io.BytesIO()
    topk_lm.save(buffer)
    buffer.seek(0)

    # Load with instantiate=False
    kwargs = shap.models.TopKLM.load(buffer, instantiate=False)
    assert "k" in kwargs
    assert kwargs["k"] == 5
    assert "batch_size" in kwargs
    assert kwargs["batch_size"] == 64
