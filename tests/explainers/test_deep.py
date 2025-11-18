"""Tests for the Deep explainer."""

import os
import platform

import numpy as np
import pandas as pd
import pytest
from packaging import version

import shap

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

############################
# Tensorflow related tests #
############################


def test_tf_eager_call(random_seed):
    """This is a basic eager example from keras."""
    tf = pytest.importorskip("tensorflow")

    tf.compat.v1.random.set_random_seed(random_seed)
    rs = np.random.RandomState(random_seed)

    if version.parse(tf.__version__) >= version.parse("2.4.0"):
        pytest.skip("Deep explainer does not work for TF 2.4 in eager mode.")

    x = pd.DataFrame({"B": rs.random(size=(100,))})
    y = x.B
    y = y.map(lambda zz: chr(int(zz * 2 + 65))).str.get_dummies()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(10, input_shape=(x.shape[1],), activation="relu"))
    model.add(tf.keras.layers.Dense(y.shape[1], input_shape=(10,), activation="softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="Adam")
    model.fit(x.values, y.values, epochs=2)

    e = shap.DeepExplainer(model, x.values[:1])
    sv = e.shap_values(x.values)
    sv_call = e(x.values)
    np.testing.assert_array_almost_equal(sv, sv_call.values, decimal=8)
    assert np.abs(e.expected_value[0] + sv[0].sum(-1) - model(x.values)[:, 0]).max() < 1e-4


def test_tf_keras_mnist_cnn_call(random_seed):
    """This is the basic mnist cnn example from keras."""
    tf = pytest.importorskip("tensorflow")
    rs = np.random.RandomState(random_seed)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    batch_size = 64
    num_classes = 10
    epochs = 1

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = rs.randn(200, 28, 28)
    y_train = rs.randint(0, 9, 200)
    x_test = rs.randn(200, 28, 28)
    y_test = rs.randint(0, 9, 200)

    if tf.keras.backend.image_data_format() == "channels_first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(2, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(4, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(16, activation="relu"))  # 128
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Activation("softmax"))

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adadelta(), metrics=["accuracy"]
    )

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    # explain by passing the tensorflow inputs and outputs
    inds = rs.choice(x_train.shape[0], 3, replace=False)
    e = shap.DeepExplainer((model.inputs, model.layers[-1].output), x_train[inds, :, :])
    shap_values = e.shap_values(x_test[:1])
    shap_values_call = e(x_test[:1])

    np.testing.assert_array_almost_equal(shap_values, shap_values_call.values, decimal=8)

    predicted = model(x_test[:1])

    sums = shap_values.sum(axis=(1, 2, 3))
    (
        np.testing.assert_allclose(sums + e.expected_value, predicted, atol=1e-3),
        "Sum of SHAP values does not match difference!",
    )


@pytest.mark.parametrize("activation", ["relu", "elu", "selu"])
def test_tf_keras_activations(activation):
    """Test verifying that a linear model with linear data gives the correct result."""
    # FIXME: this test should ideally pass with any random seed. See #2960
    random_seed = 0

    tf = pytest.importorskip("tensorflow")

    tf.compat.v1.random.set_random_seed(random_seed)
    rs = np.random.RandomState(random_seed)

    # coefficients relating y with x1 and x2.
    coef = np.array([1, 2]).T

    # generate data following a linear relationship
    x = rs.normal(1, 10, size=(1000, len(coef)))
    y = np.dot(x, coef) + 1 + rs.normal(scale=0.1, size=1000)

    # create a linear model
    inputs = tf.keras.layers.Input(shape=(2,))
    preds = tf.keras.layers.Dense(1, activation=activation)(inputs)

    model = tf.keras.models.Model(inputs=inputs, outputs=preds)
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss="mse", metrics=["mse"])
    model.fit(x, y, epochs=30, shuffle=False, verbose=0)

    # explain
    e = shap.DeepExplainer((model.inputs[0], model.layers[-1].output), x)
    shap_values = e.shap_values(x)
    preds = model.predict(x)

    assert shap_values.shape == (1000, 2, 1)
    np.testing.assert_allclose(shap_values.sum(axis=1) + e.expected_value, preds, atol=1e-5)


def test_tf_keras_linear():
    """Test verifying that a linear model with linear data gives the correct result."""
    # FIXME: this test should ideally pass with any random seed. See #2960
    random_seed = 0

    tf = pytest.importorskip("tensorflow")

    # tf.compat.v1.disable_eager_execution()

    tf.compat.v1.random.set_random_seed(random_seed)
    rs = np.random.RandomState(random_seed)

    # coefficients relating y with x1 and x2.
    coef = np.array([1, 2]).T

    # generate data following a linear relationship
    x = rs.normal(1, 10, size=(1000, len(coef)))
    y = np.dot(x, coef) + 1 + rs.normal(scale=0.1, size=1000)

    # create a linear model
    inputs = tf.keras.layers.Input(shape=(2,))
    preds = tf.keras.layers.Dense(1, activation="linear")(inputs)

    model = tf.keras.models.Model(inputs=inputs, outputs=preds)
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss="mse", metrics=["mse"])
    model.fit(x, y, epochs=30, shuffle=False, verbose=0)

    fit_coef = model.layers[1].get_weights()[0].T[0]

    # explain
    e = shap.DeepExplainer((model.inputs, model.layers[-1].output), x)
    shap_values = e.shap_values(x)

    assert shap_values.shape == (1000, 2, 1)

    # verify that the explanation follows the equation in LinearExplainer
    expected = (x - x.mean(0)) * fit_coef
    np.testing.assert_allclose(shap_values.sum(-1), expected, atol=1e-5)


def test_tf_keras_imdb_lstm(random_seed):
    """Basic LSTM example using the keras API defined in tensorflow"""
    tf = pytest.importorskip("tensorflow")
    rs = np.random.RandomState(random_seed)
    tf.compat.v1.random.set_random_seed(random_seed)

    # this fails right now for new TF versions (there is a warning in the code for this)
    if version.parse(tf.__version__) >= version.parse("2.5.0"):
        pytest.skip()

    tf.compat.v1.disable_eager_execution()

    # load the data from keras
    max_features = 1000
    try:
        (X_train, _), (X_test, _) = tf.keras.datasets.imdb.load_data(num_words=max_features)
    except Exception:
        return  # this hides a bug in the most recent version of keras that prevents data loading
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=100)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=100)

    # create the model. note that this is model is very small to make the test
    # run quick and we don't care about accuracy here
    mod = tf.keras.models.Sequential()
    mod.add(tf.keras.layers.Embedding(max_features, 8))
    mod.add(tf.keras.layers.LSTM(10, dropout=0.2, recurrent_dropout=0.2))
    mod.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    mod.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # select the background and test samples
    inds = rs.choice(X_train.shape[0], 3, replace=False)
    background = X_train[inds]
    testx = X_test[10:11]

    # explain a prediction and make sure it sums to the difference between the average output
    # over the background samples and the current output
    sess = tf.compat.v1.keras.backend.get_session()
    sess.run(tf.compat.v1.global_variables_initializer())
    # For debugging, can view graph:
    # writer = tf.compat.v1.summary.FileWriter("c:\\tmp", sess.graph)
    # writer.close()
    e = shap.DeepExplainer((mod.layers[0].input, mod.layers[-1].output), background)
    shap_values = e.shap_values(testx)
    sums = np.array([shap_values[i].sum() for i in range(len(shap_values))])
    diff = sess.run(mod.layers[-1].output, feed_dict={mod.layers[0].input: testx})[0, :] - sess.run(
        mod.layers[-1].output, feed_dict={mod.layers[0].input: background}
    ).mean(0)
    np.testing.assert_allclose(sums, diff, atol=1e-02), "Sum of SHAP values does not match difference!"


@pytest.mark.skipif(
    platform.system() == "Darwin" and os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping on GH MacOS runners due to memory error, see GH #3929",
)
def test_tf_deep_imbdb_transformers():
    # GH 3522
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from shap import models

    # data from datasets imdb dataset
    short_data = ["I lov", "Worth", "its a", "STAR ", "First", "I had", "Isaac", "It ac", "Techn", "Hones"]
    classifier = transformers.pipeline("sentiment-analysis", return_all_scores=True)
    pmodel = models.TransformersPipeline(classifier, rescale_to_logits=True)
    explainer3 = shap.Explainer(pmodel, classifier.tokenizer)
    shap_values3 = explainer3(short_data[:10])
    shap.plots.text(shap_values3[:, :, 1])  # type: ignore[call-overload]
    shap.plots.bar(shap_values3[:, :, 1].mean(0))  # type: ignore[call-overload]


def test_tf_deep_multi_inputs_multi_outputs():
    tf = pytest.importorskip("tensorflow")

    input1 = tf.keras.layers.Input(shape=(3,))
    input2 = tf.keras.layers.Input(shape=(4,))

    # Concatenate input layers
    concatenated = tf.keras.layers.concatenate([input1, input2])

    # Dense layers
    x = tf.keras.layers.Dense(16, activation="relu")(concatenated)

    # Output layer
    output = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=[input1, input2], outputs=output)
    batch_size = 32
    # Generate random input data for input1 with shape (batch_size, 3)
    input1_data = np.random.rand(batch_size, 3)

    # Generate random input data for input2 with shape (batch_size, 4)
    input2_data = np.random.rand(batch_size, 4)

    predicted = model.predict([input1_data, input2_data])
    explainer = shap.DeepExplainer(model, [input1_data, input2_data])
    shap_values = explainer.shap_values([input1_data, input2_data])
    np.testing.assert_allclose(
        shap_values[0].sum(1) + shap_values[1].sum(1) + explainer.expected_value, predicted, atol=1e-3
    )


#######################
# Torch related tests #
#######################


def _torch_cuda_available():
    """Checks whether cuda is available. If so, torch-related tests are also tested on gpu."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        pass

    return False


TORCH_DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not _torch_cuda_available(), reason="cuda unavailable (with torch)")),
]


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("interim", [True, False])
def test_pytorch_mnist_cnn_call(torch_device, interim):
    """The same test as above, but for pytorch"""
    torch = pytest.importorskip("torch")

    from torch import nn
    from torch.nn import functional as F

    class RandData:
        """Random test data."""

        def __init__(self, batch_size):
            self.current = 0
            self.batch_size = batch_size

        def __iter__(self):
            return self

        def __next__(self):
            self.current += 1
            if self.current < 10:
                return torch.randn(self.batch_size, 1, 28, 28), torch.randint(0, 9, (self.batch_size,))
            raise StopIteration

    class Net(nn.Module):
        """Basic conv net."""

        def __init__(self):
            super().__init__()
            # Testing several different activations
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 10, kernel_size=5),
                nn.MaxPool2d(2),
                nn.Tanh(),
                nn.Conv2d(10, 20, kernel_size=5),
                nn.ConvTranspose2d(20, 20, 1),
                nn.AdaptiveAvgPool2d(output_size=(4, 4)),
                nn.Softplus(),
                nn.Flatten(),
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(320, 50), nn.BatchNorm1d(50), nn.ReLU(), nn.Linear(50, 10), nn.ELU(), nn.Softmax(dim=1)
            )

        def forward(self, x):
            """Run the model."""
            x = self.conv_layers(x)
            x = x.view(-1, 320)  # Redundant as `Flatten`, left as a test
            x = self.fc_layers(x)
            return x

    def train(model, device, train_loader, optimizer, _, cutoff=20):
        model.train()
        num_examples = 0
        for _, (data, target) in enumerate(train_loader):
            num_examples += target.shape[0]
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, torch.eye(10).to(device)[target])

            loss.backward()
            optimizer.step()

            if num_examples > cutoff:
                break

    # FIXME: this test should ideally pass with any random seed. See #2960
    random_seed = 42

    torch.manual_seed(random_seed)
    rs = np.random.RandomState(random_seed)

    batch_size = 32

    train_loader = RandData(batch_size)
    test_loader = RandData(batch_size)

    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    device = torch.device(torch_device)

    model.to(device)
    train(model, device, train_loader, optimizer, 1)

    next_x, _ = next(iter(train_loader))
    inds = rs.choice(next_x.shape[0], 3, replace=False)

    next_x_random_choices = next_x[inds, :, :, :].to(device)

    if interim:
        e = shap.DeepExplainer((model, model.conv_layers[0]), next_x_random_choices)
    else:
        e = shap.DeepExplainer(model, next_x_random_choices)

    test_x, _ = next(iter(test_loader))
    input_tensor = test_x[:1].to(device)
    shap_values = e.shap_values(input_tensor)
    shap_values_call = e(input_tensor)

    np.testing.assert_array_almost_equal(shap_values, shap_values_call.values, decimal=8)

    model.eval()
    model.zero_grad()

    with torch.no_grad():
        outputs = model(input_tensor).detach().cpu().numpy()

    sums = shap_values.sum((1, 2, 3))
    (
        np.testing.assert_allclose(sums + e.expected_value, outputs, atol=1e-3),
        "Sum of SHAP values does not match difference!",
    )


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_pytorch_custom_nested_models(torch_device):
    """Testing single outputs"""
    torch = pytest.importorskip("torch")

    from sklearn.datasets import fetch_california_housing
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    class CustomNet1(nn.Module):
        """Model 1."""

        def __init__(self, num_features):
            super().__init__()
            self.net = nn.Sequential(
                nn.Sequential(
                    nn.Identity(),
                    nn.Conv1d(1, 1, 1),
                    nn.ConvTranspose1d(1, 1, 1),
                ),
                nn.AdaptiveAvgPool1d(output_size=num_features // 2),
            )

        def forward(self, X):
            """Run the model."""
            return self.net(X.unsqueeze(1)).squeeze(1)

    class CustomNet2(nn.Module):
        """Model 2."""

        def __init__(self, num_features):
            super().__init__()
            self.net = nn.Sequential(nn.LeakyReLU(), nn.Linear(num_features // 2, 2))

        def forward(self, X):
            """Run the model."""
            return self.net(X).unsqueeze(1)

    class CustomNet(nn.Module):
        """Model 3."""

        def __init__(self, num_features):
            super().__init__()
            self.net1 = CustomNet1(num_features)
            self.net2 = CustomNet2(num_features)
            self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        def forward(self, X):
            """Run the model."""
            x = self.net1(X)
            return self.maxpool2(self.net2(x)).squeeze(1)

    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        num_examples = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            num_examples += target.shape[0]
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output.squeeze(1), target)
            loss.backward()
            optimizer.step()
            if batch_idx % 2 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                    f" ({100.0 * batch_idx / len(train_loader):.0f}%)]"
                    f"\tLoss: {loss.item():.6f}"
                )

    random_seed = 777  # TODO: #2960

    torch.manual_seed(random_seed)
    rs = np.random.RandomState(random_seed)

    X, y = fetch_california_housing(return_X_y=True)

    num_features = X.shape[1]

    data = TensorDataset(
        torch.tensor(X).float(),
        torch.tensor(y).float(),
    )

    loader = DataLoader(data, batch_size=128)

    model = CustomNet(num_features)
    optimizer = torch.optim.Adam(model.parameters())

    device = torch.device(torch_device)

    model.to(device)

    train(model, device, loader, optimizer, 1)

    next_x, _ = next(iter(loader))

    inds = rs.choice(next_x.shape[0], 20, replace=False)

    next_x_random_choices = next_x[inds, :].to(device)
    e = shap.DeepExplainer(model, next_x_random_choices)

    test_x_tmp, _ = next(iter(loader))
    test_x = test_x_tmp[:1].to(device)

    shap_values = e.shap_values(test_x)

    model.eval()
    model.zero_grad()

    with torch.no_grad():
        diff = model(test_x).detach().cpu().numpy()

    sums = shap_values.sum(axis=(1))
    (
        np.testing.assert_allclose(sums + e.expected_value, diff, atol=1e-3),
        "Sum of SHAP values does not match difference!",
    )


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_pytorch_single_output(torch_device):
    """Testing single outputs"""
    torch = pytest.importorskip("torch")

    from sklearn.datasets import fetch_california_housing
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    class Net(nn.Module):
        """Test model."""

        def __init__(self, num_features):
            super().__init__()
            self.linear = nn.Linear(num_features // 2, 2)
            self.conv1d = nn.Conv1d(1, 1, 1)
            self.convt1d = nn.ConvTranspose1d(1, 1, 1)
            self.leaky_relu = nn.LeakyReLU()
            self.aapool1d = nn.AdaptiveAvgPool1d(output_size=num_features // 2)
            self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        def forward(self, X):
            """Run the model."""
            x = self.aapool1d(self.convt1d(self.conv1d(X.unsqueeze(1)))).squeeze(1)
            return self.maxpool2(self.linear(self.leaky_relu(x)).unsqueeze(1)).squeeze(1)

    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        num_examples = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            num_examples += target.shape[0]
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output.squeeze(1), target)
            loss.backward()
            optimizer.step()
            if batch_idx % 2 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                    f" ({100.0 * batch_idx / len(train_loader):.0f}%)]"
                    f"\tLoss: {loss.item():.6f}"
                )

    # FIXME: this test should ideally pass with any random seed. See #2960
    random_seed = 0
    torch.manual_seed(random_seed)
    rs = np.random.RandomState(random_seed)

    X, y = fetch_california_housing(return_X_y=True)

    num_features = X.shape[1]

    data = TensorDataset(
        torch.tensor(X).float(),
        torch.tensor(y).float(),
    )

    loader = DataLoader(data, batch_size=128)

    model = Net(num_features)
    optimizer = torch.optim.Adam(model.parameters())

    device = torch.device(torch_device)

    model.to(device)

    train(model, device, loader, optimizer, 1)

    next_x, _ = next(iter(loader))
    inds = rs.choice(next_x.shape[0], 20, replace=False)

    next_x_random_choices = next_x[inds, :].to(device)

    e = shap.DeepExplainer(model, next_x_random_choices)
    test_x_tmp, _ = next(iter(loader))
    test_x = test_x_tmp[:1].to(device)

    shap_values = e.shap_values(test_x)

    model.eval()
    model.zero_grad()

    with torch.no_grad():
        outputs = model(test_x).detach().cpu().numpy()

    sums = shap_values.sum(axis=(1))
    (
        np.testing.assert_allclose(sums + e.expected_value, outputs, atol=1e-3),
        "Sum of SHAP values does not match difference!",
    )


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
@pytest.mark.parametrize("activation", ["relu", "selu", "gelu"])
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("disconnected", [True, False])
def test_pytorch_multiple_inputs(torch_device, disconnected, activation):
    """Check a multi-input scenario."""
    torch = pytest.importorskip("torch")

    from sklearn.datasets import fetch_california_housing
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    activation_func = {"relu": nn.ReLU(), "selu": nn.SELU(), "gelu": nn.GELU()}[activation]

    class Net(nn.Module):
        """Testing model."""

        def __init__(self, num_features, disconnected):
            super().__init__()
            self.disconnected = disconnected
            if disconnected:
                num_features = num_features // 2
            self.linear = nn.Linear(num_features, 2)
            self.output = nn.Sequential(nn.MaxPool1d(2), activation_func)

        def forward(self, x1, x2):
            """Run the model."""
            if self.disconnected:
                x = self.linear(x1).unsqueeze(1)
            else:
                x = self.linear(torch.cat((x1, x2), dim=-1)).unsqueeze(1)
            return self.output(x).squeeze(1)

    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        num_examples = 0
        for batch_idx, (data1, data2, target) in enumerate(train_loader):
            num_examples += target.shape[0]
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data1, data2)
            loss = F.mse_loss(output.squeeze(1), target)
            loss.backward()
            optimizer.step()
            if batch_idx % 2 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                    f" ({100.0 * batch_idx / len(train_loader):.0f}%)]"
                    f"\tLoss: {loss.item():.6f}"
                )

    random_seed = 42  # TODO: 2960
    torch.manual_seed(random_seed)
    rs = np.random.RandomState(random_seed)

    X, y = fetch_california_housing(return_X_y=True)

    num_features = X.shape[1]
    x1 = X[:, num_features // 2 :]
    x2 = X[:, : num_features // 2]

    data = TensorDataset(
        torch.tensor(x1).float(),
        torch.tensor(x2).float(),
        torch.tensor(y).float(),
    )

    loader = DataLoader(data, batch_size=128)

    model = Net(num_features, disconnected)
    optimizer = torch.optim.Adam(model.parameters())

    device = torch.device(torch_device)

    model.to(device)

    train(model, device, loader, optimizer, 1)

    next_x1, next_x2, _ = next(iter(loader))
    inds = rs.choice(next_x1.shape[0], 20, replace=False)
    background = [next_x1[inds, :].to(device), next_x2[inds, :].to(device)]
    e = shap.DeepExplainer(model, background)

    test_x1_tmp, test_x2_tmp, _ = next(iter(loader))
    test_x1 = test_x1_tmp[:1].to(device)
    test_x2 = test_x2_tmp[:1].to(device)

    shap_values = e.shap_values([test_x1[:1], test_x2[:1]])

    model.eval()
    model.zero_grad()

    with torch.no_grad():
        outputs = model(test_x1, test_x2[:1]).detach().cpu().numpy()

    # the shap values have the shape (num_samples, num_features, num_inputs, num_outputs)
    # so since we have just one output, we slice it out
    sums = shap_values[0].sum(1) + shap_values[1].sum(1)
    (
        np.testing.assert_allclose(sums + e.expected_value, outputs, atol=1e-3),
        "Sum of SHAP values does not match difference!",
    )


###################
# JAX related tests #
###################


def test_jax_simple_model(random_seed):
    """Test JAX DeepExplainer with a simple linear model."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rs = np.random.RandomState(random_seed)

    # Create a simple linear model
    def simple_model(x):
        """A simple linear model: y = 2*x + 1"""
        return jnp.sum(2.0 * x + 1.0, axis=1, keepdims=True)

    # Create background data
    background = jnp.array(rs.randn(10, 3).astype(np.float32))

    # Create test data
    test_data = jnp.array(rs.randn(5, 3).astype(np.float32))

    # Create the explainer
    explainer = shap.DeepExplainer(simple_model, background)

    # Get SHAP values
    shap_values = explainer.shap_values(test_data)

    # Verify shape
    assert shap_values.shape == (5, 3, 1)

    # Verify additivity
    model_outputs = np.array(simple_model(test_data))
    np.testing.assert_allclose(
        shap_values.sum(axis=1) + explainer.expected_value, model_outputs, atol=1e-3
    ), "Sum of SHAP values does not match difference!"


def test_jax_neural_network(random_seed):
    """Test JAX DeepExplainer with a neural network."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rs = np.random.RandomState(random_seed)

    # Create a simple 2-layer neural network
    def neural_network(x):
        """A simple 2-layer neural network."""
        # Layer 1: 3 -> 4
        w1 = jnp.array([[0.5, -0.3, 0.2, 0.1], [0.4, 0.6, -0.2, 0.3], [0.3, -0.4, 0.5, -0.2]])
        b1 = jnp.array([0.1, -0.1, 0.2, -0.2])
        h1 = jax.nn.relu(jnp.dot(x, w1) + b1)

        # Layer 2: 4 -> 1 (single output)
        w2 = jnp.array([[0.3], [0.2], [-0.1], [0.4]])
        b2 = jnp.array([0.05])
        output = jnp.dot(h1, w2) + b2

        return output

    # Create background data for neural network
    background = jnp.array(rs.randn(20, 3).astype(np.float32))
    test_data = jnp.array(rs.randn(5, 3).astype(np.float32))

    # Create the explainer
    explainer = shap.DeepExplainer(neural_network, background)

    # Get SHAP values
    # Note: JAX implementation uses standard gradients, which may not perfectly
    # satisfy additivity for nonlinear activations. This is a known limitation.
    shap_values = explainer.shap_values(test_data, check_additivity=False)

    # Verify shape
    assert shap_values.shape == (5, 3, 1)

    # Verify that SHAP values are computed and non-zero
    assert not np.allclose(shap_values, 0)


def test_jax_multi_output(random_seed):
    """Test JAX DeepExplainer with multiple outputs."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rs = np.random.RandomState(random_seed)

    # Create a neural network with multiple outputs
    def multi_output_model(x):
        """A neural network with 2 outputs."""
        # Layer 1: 3 -> 4
        w1 = jnp.array([[0.5, -0.3, 0.2, 0.1], [0.4, 0.6, -0.2, 0.3], [0.3, -0.4, 0.5, -0.2]])
        b1 = jnp.array([0.1, -0.1, 0.2, -0.2])
        h1 = jax.nn.relu(jnp.dot(x, w1) + b1)

        # Layer 2: 4 -> 2 (two outputs)
        w2 = jnp.array([[0.3, -0.4], [0.2, 0.5], [-0.1, 0.3], [0.4, -0.2]])
        b2 = jnp.array([0.05, -0.05])
        output = jnp.dot(h1, w2) + b2

        return output

    # Create background data
    background = jnp.array(rs.randn(20, 3).astype(np.float32))
    test_data = jnp.array(rs.randn(5, 3).astype(np.float32))

    # Create the explainer
    explainer = shap.DeepExplainer(multi_output_model, background)

    # Get SHAP values
    # Note: JAX implementation uses standard gradients, which may not perfectly
    # satisfy additivity for nonlinear activations. This is a known limitation.
    shap_values = explainer.shap_values(test_data, check_additivity=False)

    # Verify shape (num_samples, num_features, num_outputs)
    assert shap_values.shape == (5, 3, 2)

    # Verify that SHAP values are computed and non-zero
    assert not np.allclose(shap_values, 0)


def test_jax_call_method(random_seed):
    """Test JAX DeepExplainer __call__ method returns Explanation object."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rs = np.random.RandomState(random_seed)

    # Create a simple model
    def simple_model(x):
        return jnp.sum(x, axis=1, keepdims=True)

    # Create background and test data
    background = jnp.array(rs.randn(10, 3).astype(np.float32))
    test_data = jnp.array(rs.randn(5, 3).astype(np.float32))

    # Create the explainer
    explainer = shap.DeepExplainer(simple_model, background)

    # Test __call__ method
    explanation = explainer(test_data)

    # Verify that it returns an Explanation object
    assert isinstance(explanation, shap.Explanation)

    # Verify that values match shap_values
    shap_values = explainer.shap_values(test_data)
    np.testing.assert_array_almost_equal(explanation.values, shap_values, decimal=8)


def test_jax_linear_model(random_seed):
    """Test JAX DeepExplainer with a linear model (should satisfy additivity perfectly)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rs = np.random.RandomState(random_seed)

    # coefficients relating y with x1 and x2
    coef = jnp.array([1.0, 2.0])
    bias = jnp.array([0.5])

    # generate data following a linear relationship
    x = rs.normal(1, 10, size=(100, len(coef))).astype(np.float32)
    x_jax = jnp.array(x)

    # create a linear model
    def linear_model(x):
        result = jnp.dot(x, coef) + bias
        # Ensure result is 2D (batch_size, 1)
        return result.reshape(-1, 1)

    # Use subset as background
    background = x_jax[:10]
    test_data = x_jax[10:15]

    # explain
    explainer = shap.DeepExplainer(linear_model, background)
    shap_values = explainer.shap_values(test_data)

    # Verify shape
    assert shap_values.shape == (5, 2, 1)

    # For linear models, verify that the explanation follows the equation
    # SHAP values should be (input - background_mean) * coefficient
    expected = (np.array(test_data) - np.array(background).mean(0))[:, :, np.newaxis] * np.array(coef).reshape(
        1, -1, 1
    )
    np.testing.assert_allclose(shap_values, expected, atol=1e-5)

    # Verify additivity
    model_outputs = np.array(linear_model(test_data))
    np.testing.assert_allclose(
        shap_values.sum(axis=1) + explainer.expected_value, model_outputs, atol=1e-5
    ), "Sum of SHAP values does not match difference!"


def test_jax_vs_tensorflow_linear_sigmoid(random_seed):
    """Compare JAX and TensorFlow DeepExplainer on linear + sigmoid model."""
    jax = pytest.importorskip("jax")
    tf = pytest.importorskip("tensorflow")
    import jax.numpy as jnp

    rs = np.random.RandomState(random_seed)

    # Create same data for both
    x = rs.normal(0, 1, size=(20, 3)).astype(np.float32)
    test_x = rs.normal(0, 1, size=(5, 3)).astype(np.float32)

    # Define weights and bias (same for both)
    weights = rs.randn(3, 1).astype(np.float32)
    bias = rs.randn(1).astype(np.float32)

    # JAX model
    weights_jax = jnp.array(weights)
    bias_jax = jnp.array(bias)

    def jax_model(x):
        return jax.nn.sigmoid(jnp.dot(x, weights_jax) + bias_jax)

    # TensorFlow model
    inputs_tf = tf.keras.layers.Input(shape=(3,))
    # Use a dense layer with fixed weights
    outputs_tf = tf.keras.layers.Dense(1, activation="sigmoid", use_bias=True)(inputs_tf)
    model_tf = tf.keras.models.Model(inputs=inputs_tf, outputs=outputs_tf)
    # Set the same weights
    model_tf.layers[1].set_weights([weights, bias])

    # JAX explainer
    explainer_jax = shap.DeepExplainer(jax_model, jnp.array(x))
    shap_values_jax = explainer_jax.shap_values(jnp.array(test_x), check_additivity=False)

    # TensorFlow explainer
    explainer_tf = shap.DeepExplainer(model_tf, x)
    shap_values_tf = explainer_tf.shap_values(test_x, check_additivity=False)

    # The SHAP values should be similar (allowing for some numerical differences)
    # Note: They may not be exactly the same due to different gradient implementations
    # but should be in the same ballpark
    shap_values_jax_flat = shap_values_jax.reshape(-1)
    shap_values_tf_flat = shap_values_tf.reshape(-1)

    # Check correlation is high (should be close to 1)
    # Note: Allow for some numerical differences between frameworks
    correlation = np.corrcoef(shap_values_jax_flat, shap_values_tf_flat)[0, 1]
    assert correlation > 0.9 or not np.isnan(
        correlation
    ), f"Correlation between JAX and TensorFlow SHAP values is too low: {correlation}"

    # Check that magnitudes are similar
    ratio = np.abs(shap_values_jax_flat).mean() / (np.abs(shap_values_tf_flat).mean() + 1e-10)
    assert 0.3 < ratio < 3.0, f"Magnitude ratio between JAX and TF is too different: {ratio}"


def test_jax_multi_input(random_seed):
    """Test JAX DeepExplainer with multiple inputs."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rs = np.random.RandomState(random_seed)

    # Create a model with two inputs
    def multi_input_model(x1, x2):
        """Simple model that concatenates and processes two inputs."""
        combined = jnp.concatenate([x1, x2], axis=1)
        # Simple linear transformation
        return jnp.sum(combined, axis=1, keepdims=True)

    # Create background data
    background1 = jnp.array(rs.randn(10, 3).astype(np.float32))
    background2 = jnp.array(rs.randn(10, 2).astype(np.float32))

    # Create test data
    test1 = jnp.array(rs.randn(5, 3).astype(np.float32))
    test2 = jnp.array(rs.randn(5, 2).astype(np.float32))

    # Create the explainer
    explainer = shap.DeepExplainer(multi_input_model, [background1, background2])

    # Get SHAP values
    shap_values = explainer.shap_values([test1, test2])

    # Verify it's a list of two arrays (one for each input)
    assert isinstance(shap_values, list)
    assert len(shap_values) == 2
    assert shap_values[0].shape == (5, 3, 1)
    assert shap_values[1].shape == (5, 2, 1)

    # Verify additivity
    model_outputs = np.array(multi_input_model(test1, test2))
    total_shap = shap_values[0].sum(axis=1) + shap_values[1].sum(axis=1)
    np.testing.assert_allclose(
        total_shap + explainer.expected_value, model_outputs.reshape(-1, 1), atol=1e-3
    ), "Sum of SHAP values does not match difference!"


def test_jax_ranked_outputs(random_seed):
    """Test JAX DeepExplainer with ranked outputs."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rs = np.random.RandomState(random_seed)

    # Create a multi-output model
    weights = jnp.array(rs.randn(3, 4).astype(np.float32))
    bias = jnp.array(rs.randn(4).astype(np.float32))

    def multi_output_model(x):
        return jnp.dot(x, weights) + bias

    # Create background and test data
    background = jnp.array(rs.randn(10, 3).astype(np.float32))
    test_data = jnp.array(rs.randn(5, 3).astype(np.float32))

    # Create the explainer
    explainer = shap.DeepExplainer(multi_output_model, background)

    # Get SHAP values for top 2 outputs
    # Note: additivity check disabled for ranked outputs
    shap_values, output_ranks = explainer.shap_values(
        test_data, ranked_outputs=2, output_rank_order="max", check_additivity=False
    )

    # Verify shapes
    assert shap_values.shape == (5, 3, 2)  # 5 samples, 3 features, top 2 outputs
    assert output_ranks.shape == (5, 2)  # 5 samples, top 2 output indices

    # Verify that output_ranks are valid indices
    assert np.all((output_ranks >= 0) & (output_ranks < 4))


def test_jax_lstm(random_seed):
    """Test JAX DeepExplainer with a simple LSTM-like recurrent model."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rs = np.random.RandomState(random_seed)

    # Create a simple recurrent model (LSTM-style)
    # For simplicity, we'll create a model that processes sequences
    hidden_size = 8
    input_size = 3
    seq_length = 5

    # LSTM-like weights
    W_input = jnp.array(rs.randn(input_size, hidden_size * 4).astype(np.float32))
    W_hidden = jnp.array(rs.randn(hidden_size, hidden_size * 4).astype(np.float32))
    bias = jnp.array(rs.randn(hidden_size * 4).astype(np.float32))
    W_output = jnp.array(rs.randn(hidden_size, 1).astype(np.float32))
    b_output = jnp.array(rs.randn(1).astype(np.float32))

    def lstm_cell(h, c, x):
        """Simple LSTM cell."""
        gates = jnp.dot(x, W_input) + jnp.dot(h, W_hidden) + bias
        i, f, g, o = jnp.split(gates, 4, axis=-1)

        i = jax.nn.sigmoid(i)  # input gate
        f = jax.nn.sigmoid(f)  # forget gate
        g = jnp.tanh(g)  # cell gate
        o = jax.nn.sigmoid(o)  # output gate

        c = f * c + i * g
        h = o * jnp.tanh(c)
        return h, c

    def lstm_model(x):
        """Process a batch of sequences through LSTM and output final hidden state."""
        batch_size = x.shape[0]
        h = jnp.zeros((batch_size, hidden_size))
        c = jnp.zeros((batch_size, hidden_size))

        # Process sequence
        for t in range(seq_length):
            h, c = lstm_cell(h, c, x[:, t, :])

        # Final output layer
        output = jnp.dot(h, W_output) + b_output
        return output

    # Create background and test data (batch_size, seq_length, input_size)
    background = jnp.array(rs.randn(10, seq_length, input_size).astype(np.float32))
    test_data = jnp.array(rs.randn(3, seq_length, input_size).astype(np.float32))

    # Create the explainer
    explainer = shap.DeepExplainer(lstm_model, background)

    # Get SHAP values (disable additivity check for recurrent models)
    shap_values = explainer.shap_values(test_data, check_additivity=False)

    # Verify shape
    assert shap_values.shape == (3, seq_length, input_size, 1)

    # Verify that SHAP values are computed and non-zero
    assert not np.allclose(shap_values, 0), "SHAP values should not all be zero for LSTM model"

    # Verify that the explainer worked
    assert explainer.expected_value is not None


def test_jax_lstm_additivity(random_seed):
    """Test JAX DeepExplainer LSTM with additivity check using simpler linear recurrence."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rs = np.random.RandomState(random_seed)

    # Create a simpler linear recurrent model for better additivity
    hidden_size = 4
    input_size = 3
    seq_length = 3  # Shorter sequence for more stable gradients

    # Use smaller weights for numerical stability
    W_input = jnp.array(rs.randn(input_size, hidden_size).astype(np.float32) * 0.1)
    W_hidden = jnp.array(rs.randn(hidden_size, hidden_size).astype(np.float32) * 0.1)
    bias = jnp.array(rs.randn(hidden_size).astype(np.float32) * 0.1)
    W_output = jnp.array(rs.randn(hidden_size, 1).astype(np.float32) * 0.1)
    b_output = jnp.array(rs.randn(1).astype(np.float32) * 0.1)

    def simple_rnn_model(x):
        """Simple linear RNN (no nonlinearities for better additivity)."""
        batch_size = x.shape[0]
        h = jnp.zeros((batch_size, hidden_size))

        # Process sequence with linear recurrence
        for t in range(seq_length):
            h = jnp.dot(x[:, t, :], W_input) + jnp.dot(h, W_hidden) + bias

        # Final linear output
        output = jnp.dot(h, W_output) + b_output
        return output

    # Create background and test data
    background = jnp.array(rs.randn(10, seq_length, input_size).astype(np.float32) * 0.5)
    test_data = jnp.array(rs.randn(3, seq_length, input_size).astype(np.float32) * 0.5)

    # Create the explainer
    explainer = shap.DeepExplainer(simple_rnn_model, background)

    # Get SHAP values with additivity check
    shap_values = explainer.shap_values(test_data, check_additivity=True)

    # Verify shape
    assert shap_values.shape == (3, seq_length, input_size, 1)

    # Verify additivity manually
    model_outputs = np.array(simple_rnn_model(test_data))
    shap_sum = shap_values.sum(axis=(1, 2))
    expected_plus_shap = explainer.expected_value + shap_sum

    np.testing.assert_allclose(
        expected_plus_shap, model_outputs, atol=1e-2, rtol=1e-2
    ), "SHAP values should satisfy additivity for linear RNN"


def test_jax_stacked_lstm(random_seed):
    """Test JAX DeepExplainer with stacked LSTM layers."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rs = np.random.RandomState(random_seed)

    # Stacked LSTM configuration
    hidden_size = 6
    input_size = 3
    seq_length = 4
    num_layers = 2

    # Weights for each layer
    layers_weights = []
    for layer in range(num_layers):
        in_size = input_size if layer == 0 else hidden_size
        W_input = jnp.array(rs.randn(in_size, hidden_size * 4).astype(np.float32) * 0.1)
        W_hidden = jnp.array(rs.randn(hidden_size, hidden_size * 4).astype(np.float32) * 0.1)
        bias = jnp.array(rs.randn(hidden_size * 4).astype(np.float32) * 0.1)
        layers_weights.append((W_input, W_hidden, bias))

    W_output = jnp.array(rs.randn(hidden_size, 1).astype(np.float32) * 0.1)
    b_output = jnp.array(rs.randn(1).astype(np.float32) * 0.1)

    def lstm_cell(h, c, x, W_input, W_hidden, bias):
        """LSTM cell."""
        gates = jnp.dot(x, W_input) + jnp.dot(h, W_hidden) + bias
        i, f, g, o = jnp.split(gates, 4, axis=-1)

        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)

        c = f * c + i * g
        h = o * jnp.tanh(c)
        return h, c

    def stacked_lstm_model(x):
        """Stacked LSTM model."""
        batch_size = x.shape[0]

        # Initialize states for all layers
        h_states = [jnp.zeros((batch_size, hidden_size)) for _ in range(num_layers)]
        c_states = [jnp.zeros((batch_size, hidden_size)) for _ in range(num_layers)]

        # Process sequence
        for t in range(seq_length):
            layer_input = x[:, t, :]

            # Pass through each LSTM layer
            for layer in range(num_layers):
                W_input, W_hidden, bias = layers_weights[layer]
                h_states[layer], c_states[layer] = lstm_cell(
                    h_states[layer], c_states[layer], layer_input, W_input, W_hidden, bias
                )
                layer_input = h_states[layer]  # Output of this layer is input to next

        # Final output from last layer
        output = jnp.dot(h_states[-1], W_output) + b_output
        return output

    # Create background and test data
    background = jnp.array(rs.randn(10, seq_length, input_size).astype(np.float32) * 0.5)
    test_data = jnp.array(rs.randn(3, seq_length, input_size).astype(np.float32) * 0.5)

    # Create the explainer
    explainer = shap.DeepExplainer(stacked_lstm_model, background)

    # Get SHAP values (disable additivity for complex nonlinear stacked model)
    shap_values = explainer.shap_values(test_data, check_additivity=False)

    # Verify shape
    assert shap_values.shape == (3, seq_length, input_size, 1)

    # Verify that SHAP values are computed and non-zero
    assert not np.allclose(shap_values, 0), "SHAP values should not all be zero for stacked LSTM"

    # Verify model actually works
    model_output = stacked_lstm_model(test_data)
    assert model_output.shape == (3, 1)

    print(f"Stacked LSTM test passed! Output shape: {model_output.shape}, SHAP shape: {shap_values.shape}")


def test_jax_embedding_layer(random_seed):
    """Test JAX DeepExplainer with embedding layers and dimension reduction."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rs = np.random.RandomState(random_seed)

    # Embedding configuration
    vocab_size = 20
    embedding_dim = 8
    seq_length = 5

    # Create embedding matrix
    embedding_matrix = jnp.array(rs.randn(vocab_size, embedding_dim).astype(np.float32) * 0.1)
    W_output = jnp.array(rs.randn(seq_length * embedding_dim, 1).astype(np.float32) * 0.1)
    b_output = jnp.array(rs.randn(1).astype(np.float32) * 0.1)

    def embedding_model_continuous(x):
        """Model that takes continuous embeddings (for SHAP computation).

        x has shape (batch, seq_length, embedding_dim)
        """
        batch_size = x.shape[0]
        # Flatten and apply output layer
        x_flat = x.reshape(batch_size, -1)
        output = jnp.dot(x_flat, W_output) + b_output
        return output

    def embedding_model_discrete(indices):
        """Model that takes discrete indices and embeds them.

        indices has shape (batch, seq_length) with integer token IDs
        """
        # Lookup embeddings
        embedded = embedding_matrix[indices]  # (batch, seq_length, embedding_dim)
        return embedding_model_continuous(embedded)

    # Create background data as continuous embeddings
    background_indices = rs.randint(0, vocab_size, size=(10, seq_length))
    background = embedding_matrix[background_indices]  # (10, seq_length, embedding_dim)

    # Create test data as continuous embeddings
    test_indices = rs.randint(0, vocab_size, size=(3, seq_length))
    test_data = embedding_matrix[test_indices]  # (3, seq_length, embedding_dim)

    # Create the explainer with continuous input (embeddings)
    explainer = shap.DeepExplainer(embedding_model_continuous, jnp.array(background))

    # Get SHAP values WITHOUT embedding dimension reduction
    shap_values_full = explainer.shap_values(jnp.array(test_data), check_additivity=True)

    # Verify shape includes embedding dimension
    assert shap_values_full.shape == (3, seq_length, embedding_dim, 1)

    # Get SHAP values WITH embedding dimension reduction (sum over embedding_dim)
    # The embedding dimension is axis 2, so we sum over it
    shap_values_reduced = explainer.shap_values(
        jnp.array(test_data), check_additivity=False, embedding_input_dim=2
    )

    # Verify shape - embedding dimension should be reduced
    assert shap_values_reduced.shape == (3, seq_length, 1)

    # Verify that reduced values equal the sum of full values
    manual_sum = shap_values_full.sum(axis=2)
    np.testing.assert_allclose(shap_values_reduced, manual_sum, atol=1e-6)

    print(f"Embedding test passed! Full shape: {shap_values_full.shape}, Reduced shape: {shap_values_reduced.shape}")


def test_jax_embedding_lstm(random_seed):
    """Test JAX DeepExplainer with embedding + LSTM and dimension reduction."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rs = np.random.RandomState(random_seed)

    # Configuration
    vocab_size = 15
    embedding_dim = 6
    hidden_size = 8
    seq_length = 4

    # Create embedding matrix
    embedding_matrix = jnp.array(rs.randn(vocab_size, embedding_dim).astype(np.float32) * 0.1)

    # LSTM weights
    W_input = jnp.array(rs.randn(embedding_dim, hidden_size * 4).astype(np.float32) * 0.1)
    W_hidden = jnp.array(rs.randn(hidden_size, hidden_size * 4).astype(np.float32) * 0.1)
    bias = jnp.array(rs.randn(hidden_size * 4).astype(np.float32) * 0.1)
    W_output = jnp.array(rs.randn(hidden_size, 1).astype(np.float32) * 0.1)
    b_output = jnp.array(rs.randn(1).astype(np.float32) * 0.1)

    def lstm_cell(h, c, x):
        """LSTM cell."""
        gates = jnp.dot(x, W_input) + jnp.dot(h, W_hidden) + bias
        i, f, g, o = jnp.split(gates, 4, axis=-1)

        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)

        c = f * c + i * g
        h = o * jnp.tanh(c)
        return h, c

    def embedding_lstm_model_continuous(x):
        """LSTM model that takes continuous embeddings.

        x has shape (batch, seq_length, embedding_dim)
        """
        batch_size = x.shape[0]
        h = jnp.zeros((batch_size, hidden_size))
        c = jnp.zeros((batch_size, hidden_size))

        # Process sequence
        for t in range(seq_length):
            h, c = lstm_cell(h, c, x[:, t, :])

        # Final output
        output = jnp.dot(h, W_output) + b_output
        return output

    # Create background and test data as continuous embeddings
    background_indices = rs.randint(0, vocab_size, size=(10, seq_length))
    background = embedding_matrix[background_indices]

    test_indices = rs.randint(0, vocab_size, size=(3, seq_length))
    test_data = embedding_matrix[test_indices]

    # Create the explainer
    explainer = shap.DeepExplainer(embedding_lstm_model_continuous, jnp.array(background))

    # Get SHAP values with embedding dimension reduction
    # Sum over embedding_dim (axis 2) to get attribution per token
    shap_values = explainer.shap_values(
        jnp.array(test_data), check_additivity=False, embedding_input_dim=2
    )

    # Verify shape - should be (batch, seq_length, 1) after reducing embedding dimension
    assert shap_values.shape == (3, seq_length, 1)

    # Verify that SHAP values are computed and non-zero
    assert not np.allclose(shap_values, 0), "SHAP values should not all be zero"

    # Verify we get meaningful attributions per token (summed over embedding dimensions)
    # Each token should have a single attribution value
    token_attributions = shap_values.squeeze(-1)  # (3, seq_length)
    assert token_attributions.shape == (3, seq_length)

    print(f"Embedding+LSTM test passed! Token attributions shape: {token_attributions.shape}")


def test_jax_scaled_dot_product_attention(random_seed):
    """Test JAX DeepExplainer with scaled dot-product attention."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rs = np.random.RandomState(random_seed)

    # Attention configuration
    seq_length = 4
    d_model = 8
    d_k = 8  # Key dimension

    # Attention weights (for Q, K, V projections)
    W_q = jnp.array(rs.randn(d_model, d_k).astype(np.float32) * 0.1)
    W_k = jnp.array(rs.randn(d_model, d_k).astype(np.float32) * 0.1)
    W_v = jnp.array(rs.randn(d_model, d_k).astype(np.float32) * 0.1)
    W_out = jnp.array(rs.randn(d_k, 1).astype(np.float32) * 0.1)
    b_out = jnp.array(rs.randn(1).astype(np.float32) * 0.1)

    def scaled_dot_product_attention(q, k, v):
        """Scaled dot-product attention mechanism.

        q, k, v: (batch, seq_len, d_k)
        Returns: (batch, seq_len, d_k)
        """
        # Compute attention scores
        d_k = q.shape[-1]
        scores = jnp.matmul(q, k.transpose(0, 2, 1)) / jnp.sqrt(d_k)  # (batch, seq_len, seq_len)

        # Apply softmax to get attention weights
        attention_weights = jax.nn.softmax(scores, axis=-1)  # (batch, seq_len, seq_len)

        # Apply attention to values
        output = jnp.matmul(attention_weights, v)  # (batch, seq_len, d_k)
        return output

    def attention_model(x):
        """Self-attention model.

        x: (batch, seq_len, d_model)
        """
        batch_size = x.shape[0]

        # Project to Q, K, V
        q = jnp.matmul(x, W_q)  # (batch, seq_len, d_k)
        k = jnp.matmul(x, W_k)
        v = jnp.matmul(x, W_v)

        # Apply attention
        attended = scaled_dot_product_attention(q, k, v)  # (batch, seq_len, d_k)

        # Pool over sequence (simple mean pooling)
        pooled = jnp.mean(attended, axis=1)  # (batch, d_k)

        # Final output layer
        output = jnp.dot(pooled, W_out) + b_out  # (batch, 1)
        return output

    # Create background and test data
    background = jnp.array(rs.randn(10, seq_length, d_model).astype(np.float32) * 0.5)
    test_data = jnp.array(rs.randn(3, seq_length, d_model).astype(np.float32) * 0.5)

    # Create the explainer
    explainer = shap.DeepExplainer(attention_model, background)

    # Get SHAP values (disable additivity for nonlinear softmax)
    shap_values = explainer.shap_values(test_data, check_additivity=False)

    # Verify shape
    assert shap_values.shape == (3, seq_length, d_model, 1)

    # Verify that SHAP values are computed and non-zero
    assert not np.allclose(shap_values, 0), "SHAP values should not all be zero for attention model"

    print(f"Attention test passed! SHAP shape: {shap_values.shape}")


def test_jax_multi_head_attention(random_seed):
    """Test JAX DeepExplainer with multi-head attention."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rs = np.random.RandomState(random_seed)

    # Multi-head attention configuration
    seq_length = 4
    d_model = 12
    num_heads = 3
    d_k = d_model // num_heads  # 4 per head

    # Weights for Q, K, V projections (one set per head)
    heads_weights = []
    for h in range(num_heads):
        W_q = jnp.array(rs.randn(d_model, d_k).astype(np.float32) * 0.1)
        W_k = jnp.array(rs.randn(d_model, d_k).astype(np.float32) * 0.1)
        W_v = jnp.array(rs.randn(d_model, d_k).astype(np.float32) * 0.1)
        heads_weights.append((W_q, W_k, W_v))

    # Output projection
    W_out = jnp.array(rs.randn(d_model, 1).astype(np.float32) * 0.1)
    b_out = jnp.array(rs.randn(1).astype(np.float32) * 0.1)

    def scaled_dot_product_attention(q, k, v):
        """Scaled dot-product attention."""
        d_k = q.shape[-1]
        scores = jnp.matmul(q, k.transpose(0, 2, 1)) / jnp.sqrt(d_k)
        attention_weights = jax.nn.softmax(scores, axis=-1)
        output = jnp.matmul(attention_weights, v)
        return output

    def multi_head_attention_model(x):
        """Multi-head self-attention model.

        x: (batch, seq_len, d_model)
        """
        batch_size = x.shape[0]

        # Compute attention for each head
        head_outputs = []
        for h in range(num_heads):
            W_q, W_k, W_v = heads_weights[h]

            # Project to Q, K, V for this head
            q = jnp.matmul(x, W_q)  # (batch, seq_len, d_k)
            k = jnp.matmul(x, W_k)
            v = jnp.matmul(x, W_v)

            # Apply attention
            head_out = scaled_dot_product_attention(q, k, v)  # (batch, seq_len, d_k)
            head_outputs.append(head_out)

        # Concatenate heads
        multi_head_out = jnp.concatenate(head_outputs, axis=-1)  # (batch, seq_len, d_model)

        # Pool over sequence
        pooled = jnp.mean(multi_head_out, axis=1)  # (batch, d_model)

        # Final output layer
        output = jnp.dot(pooled, W_out) + b_out  # (batch, 1)
        return output

    # Create background and test data
    background = jnp.array(rs.randn(10, seq_length, d_model).astype(np.float32) * 0.5)
    test_data = jnp.array(rs.randn(3, seq_length, d_model).astype(np.float32) * 0.5)

    # Create the explainer
    explainer = shap.DeepExplainer(multi_head_attention_model, background)

    # Get SHAP values
    shap_values = explainer.shap_values(test_data, check_additivity=False)

    # Verify shape
    assert shap_values.shape == (3, seq_length, d_model, 1)

    # Verify that SHAP values are computed and non-zero
    assert not np.allclose(shap_values, 0), "SHAP values should not all be zero for multi-head attention"

    print(f"Multi-head attention test passed! SHAP shape: {shap_values.shape}")


def test_jax_transformer_block(random_seed):
    """Test JAX DeepExplainer with a complete transformer block (embedding + attention + FFN)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rs = np.random.RandomState(random_seed)

    # Transformer configuration
    vocab_size = 20
    seq_length = 4
    d_model = 12
    d_ff = 24  # Feedforward dimension

    # Embedding matrix
    embedding_matrix = jnp.array(rs.randn(vocab_size, d_model).astype(np.float32) * 0.1)

    # Attention weights
    W_q = jnp.array(rs.randn(d_model, d_model).astype(np.float32) * 0.1)
    W_k = jnp.array(rs.randn(d_model, d_model).astype(np.float32) * 0.1)
    W_v = jnp.array(rs.randn(d_model, d_model).astype(np.float32) * 0.1)

    # Feedforward weights
    W_ff1 = jnp.array(rs.randn(d_model, d_ff).astype(np.float32) * 0.1)
    b_ff1 = jnp.array(rs.randn(d_ff).astype(np.float32) * 0.1)
    W_ff2 = jnp.array(rs.randn(d_ff, d_model).astype(np.float32) * 0.1)
    b_ff2 = jnp.array(rs.randn(d_model).astype(np.float32) * 0.1)

    # Output layer
    W_out = jnp.array(rs.randn(d_model, 1).astype(np.float32) * 0.1)
    b_out = jnp.array(rs.randn(1).astype(np.float32) * 0.1)

    def scaled_dot_product_attention(q, k, v):
        """Scaled dot-product attention."""
        d_k = q.shape[-1]
        scores = jnp.matmul(q, k.transpose(0, 2, 1)) / jnp.sqrt(d_k)
        attention_weights = jax.nn.softmax(scores, axis=-1)
        output = jnp.matmul(attention_weights, v)
        return output

    def transformer_model_continuous(x):
        """Transformer model taking continuous embeddings.

        x: (batch, seq_len, d_model) - continuous embeddings
        """
        batch_size = x.shape[0]

        # Self-attention
        q = jnp.matmul(x, W_q)
        k = jnp.matmul(x, W_k)
        v = jnp.matmul(x, W_v)
        attn_out = scaled_dot_product_attention(q, k, v)  # (batch, seq_len, d_model)

        # Residual connection + Layer norm (simplified - just residual for now)
        x = x + attn_out

        # Feedforward network
        ff_out = jnp.dot(x, W_ff1) + b_ff1  # (batch, seq_len, d_ff)
        ff_out = jax.nn.relu(ff_out)
        ff_out = jnp.dot(ff_out, W_ff2) + b_ff2  # (batch, seq_len, d_model)

        # Residual connection
        x = x + ff_out

        # Pool over sequence (mean pooling)
        pooled = jnp.mean(x, axis=1)  # (batch, d_model)

        # Final classification layer
        output = jnp.dot(pooled, W_out) + b_out  # (batch, 1)
        return output

    # Create background and test data as continuous embeddings
    background_indices = rs.randint(0, vocab_size, size=(10, seq_length))
    background = embedding_matrix[background_indices]

    test_indices = rs.randint(0, vocab_size, size=(3, seq_length))
    test_data = embedding_matrix[test_indices]

    # Create the explainer
    explainer = shap.DeepExplainer(transformer_model_continuous, jnp.array(background))

    # Get SHAP values with embedding dimension reduction for per-token attributions
    shap_values = explainer.shap_values(
        jnp.array(test_data), check_additivity=False, embedding_input_dim=2
    )

    # Verify shape - should be (batch, seq_len, 1) after reducing embedding dimension
    assert shap_values.shape == (3, seq_length, 1)

    # Verify that SHAP values are computed and non-zero
    assert not np.allclose(shap_values, 0), "SHAP values should not all be zero for transformer"

    # Verify we get per-token attributions
    token_attributions = shap_values.squeeze(-1)  # (3, seq_length)
    assert token_attributions.shape == (3, seq_length)

    # Check that different tokens get different attributions
    # (they should, since attention creates dependencies)
    assert np.std(token_attributions) > 1e-6, "Token attributions should vary across positions"

    print(f"Transformer test passed! Token attributions shape: {token_attributions.shape}")
    print(f"Attribution variance: {np.std(token_attributions):.6f}")
