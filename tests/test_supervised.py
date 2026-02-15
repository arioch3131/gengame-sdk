"""Tests for supervised training pipeline."""

import numpy as np
import pytest

from gengame_sdk.nn import create_network
from gengame_sdk.nn.training import TrainingConfig, train_backprop

# ============================================================
# 1. Backprop unit tests
# ============================================================


class TestBackprop:
    def test_loss_decreases(self):
        """Train a small NN on synthetic data; loss should decrease."""
        nn = create_network([4, 8, 3], activation="tanh", output_activation="softmax", seed=42)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 4))
        y = rng.integers(0, 3, size=100)

        config = TrainingConfig(learning_rate=0.05, epochs=50, batch_size=16)
        result = train_backprop(nn, X, y, config)

        assert len(result.losses) == 50
        assert result.losses[-1] < result.losses[0]
        assert result.final_loss == result.losses[-1]

    def test_perfect_fit_small_dataset(self):
        """NN should be able to memorize a tiny dataset."""
        nn = create_network([2, 16, 2], activation="relu", output_activation="softmax", seed=7)
        X = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        y = np.array([0, 0, 1, 1])  # XOR-like

        config = TrainingConfig(learning_rate=0.1, epochs=300, batch_size=4)
        result = train_backprop(nn, X, y, config)

        assert result.final_loss < 0.1

    def test_modifies_weights_in_place(self):
        """train_backprop should modify the NN's weights in place."""
        nn = create_network([3, 4, 2], activation="tanh", output_activation="softmax", seed=1)
        original_w0 = nn.weights[0].copy()

        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = np.array([0, 1])
        train_backprop(nn, X, y, TrainingConfig(epochs=5))

        assert not np.array_equal(nn.weights[0], original_w0)

    def test_single_sample(self):
        """Should handle a single training sample without errors."""
        nn = create_network([2, 3, 2], activation="sigmoid", output_activation="softmax", seed=0)
        X = np.array([[0.5, 0.5]])
        y = np.array([1])

        result = train_backprop(nn, X, y, TrainingConfig(epochs=10, batch_size=1))
        assert len(result.losses) == 10
