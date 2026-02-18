"""
Base class for board games.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from gengame_sdk.nn import NeuralNetwork


class BoardGame(ABC):
    """
    Abstract base class for board games.

    Defines the interface that all board games must implement.
    """

    @property
    @abstractmethod
    def current_player(self) -> int:
        """
        Current player to move.

        Returns:
            Player identifier (typically 1 or -1)
        """
        pass

    @property
    @abstractmethod
    def board_size(self) -> int:
        """
        Size of the board (number of cells).

        Returns:
            Number of cells on the board
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset game to initial state."""
        pass

    @abstractmethod
    def get_valid_moves(self) -> list[int]:
        """
        Get all valid moves for current player.

        Returns:
            List of valid move indices
        """
        pass

    @abstractmethod
    def make_move(self, move: int) -> bool:
        """
        Execute a move.

        Args:
            move: Move index

        Returns:
            True if move was valid and executed, False otherwise
        """
        pass

    @abstractmethod
    def check_winner(self) -> int:
        """
        Check if there's a winner.

        Returns:
            Winner player id (e.g., 1 or -1), or 0 if no winner yet
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Check if game is in terminal state (game over).

        Returns:
            True if game is over (win or draw)
        """
        pass

    @abstractmethod
    def get_board_state(self) -> list[int]:
        """
        Get current board state.

        Returns:
            List of cell values
        """
        pass

    @abstractmethod
    def get_board_for_player(self, player: int) -> list[float]:
        """
        Get board state from a player's perspective.

        Useful for neural network input where current player's pieces
        should always be represented the same way.

        Args:
            player: Player id to get perspective for

        Returns:
            Board state normalized for player (e.g., own pieces = 1, opponent = -1)
        """
        pass

    @abstractmethod
    def get_input(self, player: int, input_type: str) -> list[float]:
        """
        Get neural network input for the given player and encoding type.

        Args:
            player: Player identifier
            input_type: Input encoding type (e.g. 'board_only', 'positional', 'board_flat')

        Returns:
            Input vector for the neural network
        """
        pass

    def get_available_input_types(self) -> list[str]:
        """Return the list of supported input encoding types for this game."""
        return ["default"]

    @abstractmethod
    def copy(self) -> "BoardGame":
        """
        Create a deep copy of the game.

        Returns:
            New BoardGame instance with same state
        """
        pass

    @abstractmethod
    def display(self) -> str:
        """
        Get human-readable string representation of the board.

        Returns:
            Formatted string showing the board
        """
        pass

    def compare(self, other: "BoardGame") -> int:
        """Compare two terminal states (for parallel-resolution games).

        Returns:
            1 if self is better, -1 if other is better, 0 if equal.
        """
        s1 = self.check_winner()
        s2 = other.check_winner()
        if s1 > s2:
            return 1
        elif s1 < s2:
            return -1
        return 0

    def get_result(self) -> str | None:
        """
        Get game result as string.

        Returns:
            Result string (e.g., 'X', 'O', 'draw') or None if not terminal
        """
        if not self.is_terminal():
            return None
        winner = self.check_winner()
        if winner == 1:
            return "player1"
        elif winner == -1:
            return "player2"
        return "draw"

    @property
    def game_category(self) -> str:
        """'adversarial' (default) or 'puzzle'."""
        return "adversarial"

    @property
    def input_size(self) -> int:
        """
        Neural network input size.

        By default equals board_size, but can be overridden for
        games with additional input features.

        Returns:
            Size of neural network input vector
        """
        return self.board_size

    def enable_recording(self) -> None:
        self._recording = True

    def disable_recording(self) -> None:
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return getattr(self, "_recording", False)

    def get_record(self) -> dict[str, Any] | None:
        """Return the record of the current game. Structure defined by each game."""
        return None

    # --- Supervised training interface ---

    def get_training_methods(self) -> list[str]:
        """Supervised training methods available for this game."""
        return []

    def extract_training_data(
        self, records: list[dict[str, Any]], input_type: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract (X, y) pairs from game records.

        Args:
            records: List of game record dicts
            input_type: Input encoding type

        Returns:
            (X, y) where X is (n_samples, input_size) and y is (n_samples,) class indices
        """
        raise NotImplementedError(f"{type(self).__name__} does not support supervised training")

    def train_supervised(
        self,
        nn: "NeuralNetwork",
        records: list[dict[str, Any]],
        input_type: str,
        method: str = "backprop",
        **params,
    ) -> np.ndarray:
        """
        Train a neural network on game records. Returns modified genes (flat array).

        Args:
            nn: Neural network to train (will be copied)
            records: Game records
            input_type: Input encoding type
            method: Training method name
            **params: Training hyperparameters

        Returns:
            Flat numpy array of modified genes
        """
        raise NotImplementedError(f"{type(self).__name__} does not support supervised training")

    @property
    def output_size(self) -> int:
        """
        Neural network output size (number of possible moves).

        By default equals board_size, but can be overridden.

        Returns:
            Size of neural network output vector
        """
        return self.board_size
