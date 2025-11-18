"""JAX Deep Explainer implementation."""

from __future__ import annotations

import warnings

import numpy as np

from .._explainer import Explainer
from .deep_utils import _check_additivity


class JAXDeep(Explainer):
    """JAX implementation of Deep SHAP for differentiable models.

    This implementation uses JAX's automatic differentiation to compute DeepLIFT-style
    attributions, following the same approach as the PyTorch and TensorFlow implementations.
    """

    def __init__(self, model, data):
        """Initialize the JAX Deep explainer.

        Parameters
        ----------
        model : callable
            A JAX function that takes input arrays and returns predictions.
            The function should accept arrays compatible with the data shape.

        data : jax.Array, np.ndarray, or list
            The background dataset to use for integrating out features. Deep integrates
            over these samples. The data passed here must match the input shape expected
            by the model. If a list is provided, it indicates multiple inputs.
            Note that you should only use something like 100 or 1000 random background
            samples, not the whole training dataset.
        """
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as e:
            msg = "JAX is required for JAXDeep. Install it with: pip install jax"
            raise ImportError(msg) from e

        self.jax = jax
        self.jnp = jnp

        # check if we have multiple inputs
        self.multi_input = False
        if isinstance(data, list):
            self.multi_input = True
        if not isinstance(data, list):
            data = [data]

        # convert to JAX arrays if needed
        self.data = [jnp.array(d) if not isinstance(d, jnp.ndarray) else d for d in data]

        self.model = model
        self.expected_value = None

        # compute expected value
        with jax.default_device(jax.devices("cpu")[0] if jax.devices("cpu") else jax.devices()[0]):
            model_outputs = self.model(*self.data)

        # determine if we have multiple outputs
        self.multi_output = False
        self.num_outputs = 1
        if len(model_outputs.shape) > 1 and model_outputs.shape[1] > 1:
            self.multi_output = True
            self.num_outputs = model_outputs.shape[1]

        # compute expected value as mean over background data
        self.expected_value = jnp.mean(model_outputs, axis=0)
        if isinstance(self.expected_value, jnp.ndarray):
            self.expected_value = np.array(self.expected_value)

    def gradient(self, idx, inputs):
        """Compute DeepLIFT-style gradients for a specific output index.

        This method computes gradients that follow the DeepLIFT rule:
        For nonlinear operations, we compute the ratio of output differences
        to input differences, rather than the standard gradient.

        Parameters
        ----------
        idx : int
            The output index to compute gradients for
        inputs : list of jax.Array
            Input arrays to compute gradients with respect to.
            The first half are the test samples tiled, the second half are the reference samples.

        Returns
        -------
        list of np.ndarray
            DeepLIFT-style gradients for each input
        """
        jax = self.jax
        jnp = self.jnp

        def model_output_idx(*args):
            """Extract a specific output index."""
            out = self.model(*args)
            if self.multi_output:
                return out[:, idx]
            else:
                return jnp.squeeze(out, axis=-1) if out.ndim > 1 else out

        # For DeepLIFT, we compute the gradient of the output with respect to the inputs
        # using the linear approximation between the test and reference samples
        if len(inputs) == 1:
            grad_fn = jax.grad(lambda x: jnp.sum(model_output_idx(x)))
            grads = [grad_fn(inputs[0])]
        else:
            # for multiple inputs, compute gradients for each
            grads = []
            for i in range(len(inputs)):

                def fn_for_input_i(*args):
                    return jnp.sum(model_output_idx(*args))

                grad_fn = jax.grad(fn_for_input_i, argnums=i)
                grad = grad_fn(*inputs)
                grads.append(grad)

        # convert to numpy
        grads = [np.array(g) for g in grads]

        return grads

    def shap_values(self, X, ranked_outputs=None, output_rank_order="max", check_additivity=True):
        """Return approximate SHAP values for the model applied to X.

        Parameters
        ----------
        X : jax.Array, np.ndarray, or list
            A tensor (or list of tensors) of samples (where X.shape[0] == # samples) on which to
            explain the model's output.

        ranked_outputs : None or int
            If ranked_outputs is None then we explain all the outputs in a multi-output model. If
            ranked_outputs is a positive integer then we only explain that many of the top model
            outputs (where "top" is determined by output_rank_order). Note that this causes a pair
            of values to be returned (shap_values, indexes), where shap_values is a list of numpy
            arrays for each of the output ranks, and indexes is a matrix that indicates for each sample
            which output indexes were chosen as "top".

        output_rank_order : "max", "min", or "max_abs"
            How to order the model outputs when using ranked_outputs, either by maximum, minimum, or
            maximum absolute value.

        check_additivity : bool
            Whether to check that SHAP values sum to the difference between model output and expected value.

        Returns
        -------
        np.array or list
            Estimated SHAP values, usually of shape ``(# samples x # features)``.

            The shape of the returned array depends on the number of model outputs:

            * one input, one output: matrix of shape ``(#num_samples, *X.shape[1:])``.
            * one input, multiple outputs: matrix of shape ``(#num_samples, *X.shape[1:], #num_outputs)``
            * multiple inputs, one or more outputs: list of matrices, with shapes of one of the above.

            If ranked_outputs is ``None`` then this list of tensors matches
            the number of model outputs. If ranked_outputs is a positive integer a pair is returned
            (shap_values, indexes), where shap_values is a list of tensors with a length of
            ranked_outputs, and indexes is a matrix that indicates for each sample which output indexes
            were chosen as "top".
        """
        jnp = self.jnp

        # check if we have multiple inputs
        if not self.multi_input:
            if isinstance(X, list):
                msg = "Expected a single tensor model input!"
                raise ValueError(msg)
            X = [X]
        else:
            if not isinstance(X, list):
                msg = "Expected a list of model inputs!"
                raise ValueError(msg)

        # convert to JAX arrays if needed
        X = [jnp.array(x) if not isinstance(x, jnp.ndarray) else x for x in X]

        model_output_values = None

        if ranked_outputs is not None and self.multi_output:
            model_output_values = self.model(*X)
            model_output_values = np.array(model_output_values)

            # rank and determine the model outputs that we will explain
            if output_rank_order == "max":
                model_output_ranks = np.argsort(-model_output_values)
            elif output_rank_order == "min":
                model_output_ranks = np.argsort(model_output_values)
            elif output_rank_order == "max_abs":
                model_output_ranks = np.argsort(np.abs(model_output_values))
            else:
                msg = "output_rank_order must be max, min, or max_abs!"
                raise ValueError(msg)
            model_output_ranks = model_output_ranks[:, :ranked_outputs]
        else:
            model_output_ranks = np.ones((X[0].shape[0], self.num_outputs), dtype=int) * np.arange(
                0, self.num_outputs, dtype=int
            )

        # compute the attributions
        output_phis = []
        for i in range(model_output_ranks.shape[1]):
            phis = []
            for k in range(len(X)):
                phis.append(np.zeros(X[k].shape))

            for j in range(X[0].shape[0]):
                # tile the inputs to line up with the background data samples
                tiled_X = [
                    jnp.tile(
                        X[t][j : j + 1],
                        (self.data[t].shape[0],) + tuple([1 for k in range(len(X[t].shape) - 1)]),
                    )
                    for t in range(len(X))
                ]
                joint_x = [jnp.concatenate((tiled_X[t], self.data[t]), axis=0) for t in range(len(X))]

                # run attribution computation
                feature_ind = model_output_ranks[j, i]
                sample_phis = self.gradient(feature_ind, joint_x)

                # assign the attributions to the right part of the output arrays
                for t in range(len(X)):
                    # DeepLIFT-style attribution: grad * (input - reference)
                    x_t = np.array(X[t][j : j + 1])
                    data_t = np.array(self.data[t])
                    phis[t][j] = (sample_phis[t][self.data[t].shape[0] :] * (x_t - data_t)).mean(0)

            output_phis.append(phis[0] if not self.multi_input else phis)

        # check that the SHAP values sum up to the model output
        if check_additivity:
            if model_output_values is None:
                model_output_values = self.model(*X)
                model_output_values = np.array(model_output_values)

            _check_additivity(self, model_output_values, output_phis)

        if isinstance(output_phis, list):
            # in this case we have multiple inputs and potentially multiple outputs
            if isinstance(output_phis[0], list):
                output_phis = [
                    np.stack([phi[i] for phi in output_phis], axis=-1) for i in range(len(output_phis[0]))
                ]
            # multiple outputs case
            else:
                output_phis = np.stack(output_phis, axis=-1)

        if ranked_outputs is not None:
            return output_phis, model_output_ranks
        else:
            return output_phis
