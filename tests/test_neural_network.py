"""Tests for the NeuralNetwork module."""

import numpy as np
import pytest

from gengame_sdk.nn.network import (
    NeuralNetwork,
    _apply_activation,
    _initialize_weights,
    create_network,
)


class TestActivations:
    """Tests for activation functions."""

    def test_tanh(self):
        x = np.array([0.0, 1.0, -1.0])
        result = _apply_activation(x, "tanh")
        np.testing.assert_allclose(result, np.tanh(x))

    def test_relu(self):
        x = np.array([-2.0, 0.0, 3.0])
        result = _apply_activation(x, "relu")
        np.testing.assert_array_equal(result, [0.0, 0.0, 3.0])

    def test_sigmoid(self):
        x = np.array([0.0])
        result = _apply_activation(x, "sigmoid")
        assert result[0] == pytest.approx(0.5)

    def test_sigmoid_clip_extreme(self):
        """Sigmoid should not overflow with extreme values."""
        x = np.array([1000.0, -1000.0])
        result = _apply_activation(x, "sigmoid")
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.0)

    def test_leaky_relu(self):
        x = np.array([-10.0, 0.0, 5.0])
        result = _apply_activation(x, "leaky_relu")
        np.testing.assert_allclose(result, [-0.1, 0.0, 5.0])

    def test_softmax(self):
        x = np.array([1.0, 2.0, 3.0])
        result = _apply_activation(x, "softmax")
        assert result.sum() == pytest.approx(1.0)
        assert all(r > 0 for r in result)
        # Order preserved
        assert result[2] > result[1] > result[0]

    def test_linear(self):
        x = np.array([1.5, -2.5])
        result = _apply_activation(x, "linear")
        np.testing.assert_array_equal(result, x)

    def test_unknown_activation(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            _apply_activation(np.array([1.0]), "unknown")


class TestInitializeWeights:
    """Tests for weight initialization."""

    def test_xavier_shape(self):
        w = _initialize_weights(4, 3, "xavier")
        assert w.shape == (4, 3)

    def test_he_shape(self):
        w = _initialize_weights(10, 5, "he")
        assert w.shape == (10, 5)

    def test_uniform_range(self):
        rng = np.random.default_rng(42)
        w = _initialize_weights(100, 100, "uniform", rng)
        assert w.min() >= -0.5
        assert w.max() <= 0.5

    def test_normal_distribution(self):
        rng = np.random.default_rng(42)
        w = _initialize_weights(100, 100, "normal", rng)
        # Mean ~0, std ~0.1
        assert abs(w.mean()) < 0.05
        assert abs(w.std() - 0.1) < 0.05

    def test_reproducibility(self):
        w1 = _initialize_weights(4, 3, "xavier", np.random.default_rng(42))
        w2 = _initialize_weights(4, 3, "xavier", np.random.default_rng(42))
        np.testing.assert_array_equal(w1, w2)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown initialization"):
            _initialize_weights(4, 3, "unknown")


class TestNeuralNetwork:
    """Tests for NeuralNetwork."""

    def test_initialize_creates_weights(self):
        nn = NeuralNetwork(architecture=[4, 8, 2])
        assert not nn.is_initialized
        nn.initialize(seed=42)
        assert nn.is_initialized
        assert len(nn.weights) == 2
        assert len(nn.biases) == 2

    def test_weight_shapes(self):
        nn = NeuralNetwork(architecture=[9, 32, 16, 9]).initialize(seed=0)
        assert nn.weights[0].shape == (9, 32)
        assert nn.weights[1].shape == (32, 16)
        assert nn.weights[2].shape == (16, 9)
        assert nn.biases[0].shape == (32,)
        assert nn.biases[1].shape == (16,)
        assert nn.biases[2].shape == (9,)

    def test_forward_output_shape(self):
        nn = NeuralNetwork(architecture=[9, 32, 9]).initialize(seed=42)
        output = nn.forward(np.zeros(9))
        assert output.shape == (9,)

    def test_forward_not_initialized(self):
        nn = NeuralNetwork(architecture=[4, 2])
        with pytest.raises(RuntimeError, match="not initialized"):
            nn.forward(np.zeros(4))

    def test_forward_reproducible(self):
        nn = NeuralNetwork(architecture=[4, 8, 2]).initialize(seed=42)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        out1 = nn.forward(x)
        out2 = nn.forward(x)
        np.testing.assert_array_equal(out1, out2)

    def test_output_activation(self):
        nn = NeuralNetwork(
            architecture=[4, 8, 3],
            activation="relu",
            output_activation="softmax",
        ).initialize(seed=42)
        output = nn.forward(np.ones(4))
        assert output.sum() == pytest.approx(1.0)
        assert all(o >= 0 for o in output)

    def test_default_output_activation(self):
        """output_activation defaults to activation."""
        nn = NeuralNetwork(architecture=[2, 2], activation="relu")
        assert nn.output_activation == "relu"

    def test_to_flat_from_flat_roundtrip(self):
        nn = NeuralNetwork(architecture=[9, 32, 9], activation="tanh").initialize(seed=42)
        flat = nn.to_flat()
        nn2 = NeuralNetwork.from_flat(flat, [9, 32, 9], activation="tanh")
        x = np.ones(9)
        np.testing.assert_allclose(nn.forward(x), nn2.forward(x))

    def test_to_flat_not_initialized(self):
        nn = NeuralNetwork(architecture=[4, 2])
        with pytest.raises(RuntimeError, match="not initialized"):
            nn.to_flat()

    def test_num_parameters(self):
        nn = NeuralNetwork(architecture=[9, 32, 9])
        # Not initialized: theoretical calculation
        expected = 9 * 32 + 32 + 32 * 9 + 9  # = 288 + 32 + 288 + 9 = 617
        assert nn.num_parameters == expected
        # Initialized: same result
        nn.initialize(seed=0)
        assert nn.num_parameters == expected

    def test_copy_is_independent(self):
        nn = NeuralNetwork(architecture=[4, 8, 2]).initialize(seed=42)
        nn_copy = nn.copy()
        # Modifying the copy should not change the original
        nn_copy.weights[0][0, 0] = 999.0
        assert nn.weights[0][0, 0] != 999.0

    def test_repr(self):
        nn = NeuralNetwork(architecture=[4, 2])
        assert "not initialized" in repr(nn)
        nn.initialize()
        assert "initialized" in repr(nn)


class TestCreateNetwork:
    """Tests for the factory function."""

    def test_creates_initialized_network(self):
        nn = create_network([4, 8, 2], seed=42)
        assert nn.is_initialized
        assert nn.architecture == [4, 8, 2]

    def test_all_params(self):
        nn = create_network(
            [9, 16, 9],
            activation="relu",
            output_activation="softmax",
            init_method="he",
            seed=0,
        )
        assert nn.activation == "relu"
        assert nn.output_activation == "softmax"
        assert nn.init_method == "he"

    def test_reproducibility(self):
        nn1 = create_network([4, 8, 2], seed=42)
        nn2 = create_network([4, 8, 2], seed=42)
        np.testing.assert_array_equal(nn1.to_flat(), nn2.to_flat())
