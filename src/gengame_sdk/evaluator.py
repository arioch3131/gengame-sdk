"""
Base class for game evaluators/opponents.
"""

from abc import ABC, abstractmethod

from .game import BoardGame


class Evaluator(ABC):
    """Base class for game evaluators/opponents."""

    @abstractmethod
    def get_move(self, game: BoardGame) -> int:
        """
        Get next move for current player.

        Args:
            game: Current game state

        Returns:
            Move index to play
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Evaluator name."""
        ...
