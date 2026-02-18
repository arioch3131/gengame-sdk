"""
Generic neural network evaluator for any BoardGame.
"""

import numpy as np

from .evaluator import Evaluator
from .game import BoardGame


class NeuralNetworkEvaluator(Evaluator):
    """
    Neural network evaluator that works with any BoardGame.

    Uses game.get_input() for encoding and game.output_size for masking.
    """

    def __init__(self, network, input_type: str = "default"):
        self.network = network
        self.input_type = input_type

    @property
    def name(self) -> str:
        return "neural"

    def get_move(self, game: BoardGame) -> int:
        nn_input = np.array(game.get_input(game.current_player, self.input_type))
        output = self.network.forward(nn_input)

        masked = np.full(game.output_size, -np.inf)
        for v in game.get_valid_moves():
            masked[v] = output[v]

        return int(np.argmax(masked))
