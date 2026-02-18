"""Tests for NeuralNetworkEvaluator."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from gengame_sdk.nn_evaluator import NeuralNetworkEvaluator


class _MockGame:
    """Minimal mock game for testing."""

    def __init__(self, valid_moves=None, board=None):
        self._valid_moves = valid_moves or list(range(9))
        self._board = board or [0.0] * 9
        self._current_player = 1

    @property
    def current_player(self):
        return self._current_player

    @property
    def output_size(self):
        return 9

    def get_valid_moves(self):
        return self._valid_moves

    def get_input(self, player, input_type):
        return self._board


class TestNeuralNetworkEvaluator:
    """Test NeuralNetworkEvaluator."""

    def test_name(self):
        ev = NeuralNetworkEvaluator(network=None)
        assert ev.name == "neural"

    def test_get_move_uses_network(self):
        game = _MockGame()
        mock_network = MagicMock()
        output = np.zeros(9)
        output[4] = 1.0
        mock_network.forward.return_value = output

        ev = NeuralNetworkEvaluator(mock_network, input_type="board_only")
        move = ev.get_move(game)
        assert move == 4
        mock_network.forward.assert_called_once()

    def test_masks_invalid_moves(self):
        # Only moves 0 and 1 are valid; network prefers move 4
        game = _MockGame(valid_moves=[0, 1])
        mock_network = MagicMock()
        output = np.zeros(9)
        output[4] = 10.0
        output[1] = 1.0
        mock_network.forward.return_value = output

        ev = NeuralNetworkEvaluator(mock_network, input_type="board_only")
        move = ev.get_move(game)
        assert move == 1  # Best valid move

    def test_single_valid_move(self):
        game = _MockGame(valid_moves=[7])
        mock_network = MagicMock()
        mock_network.forward.return_value = np.zeros(9)

        ev = NeuralNetworkEvaluator(mock_network, input_type="board_only")
        move = ev.get_move(game)
        assert move == 7
