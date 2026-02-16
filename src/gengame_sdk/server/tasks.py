"""
Task definitions for game evaluation.

Tasks are serializable dataclasses that can be sent to workers.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Task:
    """Base class for all tasks."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class Result:
    """Base class for all results."""

    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================================
# EVALUATION TASKS
# =============================================================================


@dataclass
class EvalTask(Task):
    """
    Task to evaluate a neural network by playing games.

    Attributes:
        genes: Flattened neural network weights
        architecture: Network architecture
        activation: Hidden layer activation
        output_activation: Output layer activation
        game_type: Type of game ('tictactoe', etc.)
        input_type: Input encoding type ('board_only', 'positional', 'historical')
        opponent: Opponent type ('random', 'minimax', etc.)
        num_games: Number of games to play
        opponent_mix: Optional dict of opponent probabilities
    """

    genes: list[float]
    architecture: list[int]
    activation: str = "tanh"
    output_activation: str = "softmax"
    game_type: str = "tictactoe"
    input_type: str = "board_only"
    opponent: str = "random"
    num_games: int = 50
    opponent_mix: dict[str, float] | None = None
    record_games: bool = False
    hof_opponents: list[list[float]] | None = None


@dataclass
class EvalResult(Result):
    """Result of an evaluation task."""

    fitness: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    games_played: int = 0
    game_records: list[dict[str, Any]] | None = None


# =============================================================================
# SUPERVISED TRAINING TASKS
# =============================================================================


@dataclass
class SupervisedTask(Task):
    """Task for supervised training on game records."""

    genes: list[float] = field(default_factory=list)
    architecture: list[int] = field(default_factory=list)
    activation: str = "tanh"
    output_activation: str = "softmax"
    game_type: str = "tictactoe"
    input_type: str = "board_only"
    method: str = "backprop"
    records: list[dict[str, Any]] = field(default_factory=list)
    training_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SupervisedResult(Result):
    """Result of supervised training."""

    genes: list[float] | None = None
    losses: list[float] = field(default_factory=list)
    final_loss: float = 0.0


# =============================================================================
# MATCH TASKS (TOURNAMENTS)
# =============================================================================


@dataclass
class MatchTask(Task):
    """
    Task to play a match between two neural networks.

    A match consists of multiple games with alternating first player.
    """

    player1_genes: list[float] = field(default_factory=list)
    player1_name: str = ""
    player2_genes: list[float] = field(default_factory=list)
    player2_name: str = ""
    architecture: list[int] = field(default_factory=list)
    activation: str = "tanh"
    output_activation: str = "softmax"
    game_type: str = "tictactoe"
    input_type: str = "board_only"
    num_games: int = 10
    seed: int = 0
    # Per-player overrides (None = use shared fields above)
    player2_architecture: list[int] | None = None
    player2_activation: str | None = None
    player2_output_activation: str | None = None
    player2_input_type: str | None = None


@dataclass
class MatchResult(Result):
    """Result of a match between two players."""

    player1_wins: int = 0
    player2_wins: int = 0
    draws: int = 0
    game_records: list[dict[str, Any]] | None = None


# =============================================================================
# BATCH TASKS
# =============================================================================


@dataclass
class BatchEvalTask(Task):
    """Batch of evaluation tasks for efficiency."""

    tasks: list[EvalTask] = field(default_factory=list)


@dataclass
class BatchEvalResult(Result):
    """Results of batch evaluation."""

    results: list[EvalResult] = field(default_factory=list)
