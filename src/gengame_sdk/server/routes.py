"""
FastAPI routes for the Game API.

Reuses execution functions from the service module.
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor

from fastapi import APIRouter, HTTPException

from .play import SessionManager
from .schemas import (
    BatchEvalRequest,
    BatchEvalResponse,
    EvalRequest,
    EvalResponse,
    GameInfoResponse,
    MatchRequest,
    MatchResponse,
    PlayMoveRequest,
    PlayMoveResponse,
    PlayStartRequest,
    PlayStartResponse,
    SupervisedRequest,
    SupervisedResponse,
)
from .service import (
    execute_eval_task,
    execute_match_task,
    execute_supervised_task,
)
from .tasks import EvalTask, MatchTask, SupervisedTask

router = APIRouter()
session_manager = SessionManager()

# Process pool for CPU-bound evaluations
_executor: ProcessPoolExecutor | None = None


def get_executor() -> ProcessPoolExecutor:
    """Return the global ProcessPoolExecutor."""
    global _executor  # noqa: PLW0603
    if _executor is None:
        import os

        _executor = ProcessPoolExecutor(max_workers=os.cpu_count() or 4)
    return _executor


def set_executor(executor: ProcessPoolExecutor) -> None:
    """Set the ProcessPoolExecutor (called by create_app)."""
    global _executor  # noqa: PLW0603
    _executor = executor


def shutdown_executor() -> None:
    """Shutdown the ProcessPoolExecutor."""
    global _executor  # noqa: PLW0603
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None


# =============================================================================
# EVALUATE
# =============================================================================


@router.post("/evaluate", response_model=EvalResponse)
async def evaluate(request: EvalRequest) -> EvalResponse:
    """Evaluate a neural network by playing games."""
    task = EvalTask(
        genes=request.genes,
        architecture=request.architecture,
        activation=request.activation,
        output_activation=request.output_activation,
        game_type=request.game_type,
        input_type=request.input_type,
        opponent=request.opponent,
        num_games=request.num_games,
        opponent_mix=request.opponent_mix,
        record_games=request.record_games,
    )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(get_executor(), execute_eval_task, task)

    return EvalResponse(
        success=result.success,
        error=result.error,
        fitness=result.fitness,
        metrics=result.metrics,
        games_played=result.games_played,
        game_records=result.game_records,
    )


@router.post("/evaluate/batch", response_model=BatchEvalResponse)
async def evaluate_batch(request: BatchEvalRequest) -> BatchEvalResponse:
    """Evaluate a batch of neural networks in parallel."""
    tasks = [
        EvalTask(
            genes=req.genes,
            architecture=req.architecture,
            activation=req.activation,
            output_activation=req.output_activation,
            game_type=req.game_type,
            input_type=req.input_type,
            opponent=req.opponent,
            num_games=req.num_games,
            opponent_mix=req.opponent_mix,
            record_games=req.record_games,
        )
        for req in request.tasks
    ]

    loop = asyncio.get_event_loop()
    futures = [loop.run_in_executor(get_executor(), execute_eval_task, t) for t in tasks]
    results = await asyncio.gather(*futures)

    responses = [
        EvalResponse(
            success=r.success,
            error=r.error,
            fitness=r.fitness,
            metrics=r.metrics,
            games_played=r.games_played,
            game_records=r.game_records,
        )
        for r in results
    ]

    return BatchEvalResponse(success=True, results=responses)


# =============================================================================
# MATCH
# =============================================================================


@router.post("/match", response_model=MatchResponse)
async def match(request: MatchRequest) -> MatchResponse:
    """Play a match between two neural networks."""
    task = MatchTask(
        player1_genes=request.player1_genes,
        player1_name=request.player1_name,
        player2_genes=request.player2_genes,
        player2_name=request.player2_name,
        architecture=request.architecture,
        activation=request.activation,
        output_activation=request.output_activation,
        game_type=request.game_type,
        input_type=request.input_type,
        num_games=request.num_games,
        seed=request.seed,
        player2_architecture=request.player2_architecture,
        player2_activation=request.player2_activation,
        player2_output_activation=request.player2_output_activation,
        player2_input_type=request.player2_input_type,
    )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(get_executor(), execute_match_task, task)

    return MatchResponse(
        success=result.success,
        error=result.error,
        player1_wins=result.player1_wins,
        player2_wins=result.player2_wins,
        draws=result.draws,
        game_records=result.game_records,
    )


# =============================================================================
# SUPERVISED
# =============================================================================


@router.post("/supervised", response_model=SupervisedResponse)
async def supervised(request: SupervisedRequest) -> SupervisedResponse:
    """Execute supervised training."""
    task = SupervisedTask(
        genes=request.genes,
        architecture=request.architecture,
        activation=request.activation,
        output_activation=request.output_activation,
        game_type=request.game_type,
        input_type=request.input_type,
        method=request.method,
        records=request.records,
        training_params=request.training_params,
    )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(get_executor(), execute_supervised_task, task)

    return SupervisedResponse(
        success=result.success,
        error=result.error,
        genes=result.genes,
        losses=result.losses,
        final_loss=result.final_loss,
    )


# =============================================================================
# PLAY
# =============================================================================


@router.post("/play/start", response_model=PlayStartResponse)
async def play_start(request: PlayStartRequest) -> PlayStartResponse:
    """Start an interactive play session."""
    try:
        session_id, session = session_manager.create_session(
            game_type=request.game_type,
            genes=request.genes,
            architecture=request.architecture,
            activation=request.activation,
            output_activation=request.output_activation,
            input_type=request.input_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    game = session.game

    # If AI plays first, make its move
    ai_move = None
    if game.current_player == session.ai_player:
        ai_move = session.nn_evaluator.get_move(game)
        game.make_move(ai_move)

    return PlayStartResponse(
        session_id=session_id,
        board=game.get_board_state(),
        board_display=game.display(),
        current_player=game.current_player,
        valid_moves=game.get_valid_moves() if not game.is_terminal() else [],
        human_player=session.human_player,
        ai_player=session.ai_player,
    )


@router.post("/play/move", response_model=PlayMoveResponse)
async def play_move(request: PlayMoveRequest) -> PlayMoveResponse:
    """Play a human move and get AI response."""
    session = session_manager.get_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    game = session.game

    if game.is_terminal():
        raise HTTPException(status_code=400, detail="Game is already over")

    # Validate and play human move
    valid_moves = game.get_valid_moves()
    if request.move not in valid_moves:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid move: {request.move}. Valid moves: {valid_moves}",
        )

    game.make_move(request.move)

    # AI move if game is not over
    ai_move = None
    if not game.is_terminal() and game.current_player == session.ai_player:
        ai_move = session.nn_evaluator.get_move(game)
        game.make_move(ai_move)

    winner = None
    if game.is_terminal():
        w = game.check_winner()
        winner = w if w != 0 else 0

    return PlayMoveResponse(
        board=game.get_board_state(),
        board_display=game.display(),
        current_player=game.current_player if not game.is_terminal() else 0,
        valid_moves=game.get_valid_moves() if not game.is_terminal() else [],
        is_terminal=game.is_terminal(),
        winner=winner,
        ai_move=ai_move,
    )


# =============================================================================
# INFO
# =============================================================================


def _get_game_info(game_type: str) -> GameInfoResponse:
    """Build game info response."""
    from gengame_sdk.plugin import get_game_plugin

    plugin = get_game_plugin(game_type)
    game = plugin.game_class()

    return GameInfoResponse(
        game_type=game_type,
        game_category=game.game_category,
        board_size=game.board_size,
        output_size=game.output_size,
        available_evaluators=list(plugin.evaluators.keys()),
    )
