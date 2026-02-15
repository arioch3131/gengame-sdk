"""
Pydantic schemas for the Game API.

Mirrors task dataclasses with Pydantic validation for HTTP exchanges.
"""

from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# EVALUATION
# =============================================================================


class EvalRequest(BaseModel):
    """Neural network evaluation request."""

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


class EvalResponse(BaseModel):
    """Evaluation response."""

    success: bool = True
    error: str | None = None
    fitness: float = 0.0
    metrics: dict[str, Any] = Field(default_factory=dict)
    games_played: int = 0
    game_records: list[dict[str, Any]] | None = None


class BatchEvalRequest(BaseModel):
    """Batch evaluation request."""

    tasks: list[EvalRequest]


class BatchEvalResponse(BaseModel):
    """Batch evaluation response."""

    success: bool = True
    error: str | None = None
    results: list[EvalResponse] = Field(default_factory=list)


# =============================================================================
# MATCH (TOURNAMENT)
# =============================================================================


class MatchRequest(BaseModel):
    """Match request between two neural networks."""

    player1_genes: list[float] = Field(default_factory=list)
    player1_name: str = ""
    player2_genes: list[float] = Field(default_factory=list)
    player2_name: str = ""
    architecture: list[int] = Field(default_factory=list)
    activation: str = "tanh"
    output_activation: str = "softmax"
    game_type: str = "tictactoe"
    input_type: str = "board_only"
    num_games: int = 10
    seed: int = 0
    player2_architecture: list[int] | None = None
    player2_activation: str | None = None
    player2_output_activation: str | None = None
    player2_input_type: str | None = None


class MatchResponse(BaseModel):
    """Match response."""

    success: bool = True
    error: str | None = None
    player1_wins: int = 0
    player2_wins: int = 0
    draws: int = 0
    game_records: list[dict[str, Any]] | None = None


# =============================================================================
# SUPERVISED TRAINING
# =============================================================================


class SupervisedRequest(BaseModel):
    """Supervised training request."""

    genes: list[float] = Field(default_factory=list)
    architecture: list[int] = Field(default_factory=list)
    activation: str = "tanh"
    output_activation: str = "softmax"
    game_type: str = "tictactoe"
    input_type: str = "board_only"
    method: str = "backprop"
    records: list[dict[str, Any]] = Field(default_factory=list)
    training_params: dict[str, Any] = Field(default_factory=dict)


class SupervisedResponse(BaseModel):
    """Supervised training response."""

    success: bool = True
    error: str | None = None
    genes: list[float] | None = None
    losses: list[float] = Field(default_factory=list)
    final_loss: float = 0.0


# =============================================================================
# PLAY (INTERACTIVE)
# =============================================================================


class PlayStartRequest(BaseModel):
    """Interactive play session start request."""

    game_type: str = "tictactoe"
    genes: list[float] = Field(default_factory=list)
    architecture: list[int] = Field(default_factory=list)
    activation: str = "tanh"
    output_activation: str = "softmax"
    input_type: str = "board_only"


class PlayStartResponse(BaseModel):
    """Play session start response."""

    session_id: str
    board: list[int]
    board_display: str
    current_player: int
    valid_moves: list[int]
    human_player: int
    ai_player: int


class PlayMoveRequest(BaseModel):
    """Play session move request."""

    session_id: str
    move: int


class PlayMoveResponse(BaseModel):
    """Play session move response."""

    board: list[int]
    board_display: str
    current_player: int
    valid_moves: list[int]
    is_terminal: bool
    winner: int | None = None
    ai_move: int | None = None


# =============================================================================
# INFO
# =============================================================================


class GameInfoResponse(BaseModel):
    """Game info response."""

    game_type: str
    game_category: str
    board_size: int
    output_size: int
    available_evaluators: list[str] = Field(default_factory=list)
