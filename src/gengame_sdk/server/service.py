"""
Game execution service for evaluating neural networks.
"""

import random
from typing import Any

import numpy as np

from gengame_sdk.game import BoardGame
from gengame_sdk.nn import NeuralNetwork
from gengame_sdk.nn_evaluator import NeuralNetworkEvaluator
from gengame_sdk.plugin import get_game_plugin

from .tasks import EvalResult, EvalTask, MatchResult, MatchTask, SupervisedResult, SupervisedTask

# Default shuffle range for puzzle training
DEFAULT_SHUFFLE_RANGE = (15, 25)
# Shuffle range for tournament matches (medium-hard)
TOURNAMENT_SHUFFLE_RANGE = (15, 30)


def _get_game(game_type: str) -> BoardGame:
    """Factory to instantiate a game by type name."""
    return get_game_plugin(game_type).game_class()


def _get_nn_evaluator(network, input_type: str):
    """Create a generic NeuralNetworkEvaluator."""
    return NeuralNetworkEvaluator(network, input_type=input_type)


def _get_evaluator(game_type: str, name: str):
    """Factory to get an evaluator by game type and name."""
    return get_game_plugin(game_type).get_evaluator(name)


def _solve_puzzle(solver, game: BoardGame) -> None:
    """Run a solver (NN evaluator or PuzzleSolver) on a puzzle until terminal."""
    while not game.is_terminal():
        move = solver.get_move(game)
        game.make_move(move)


def _play_puzzle_game(
    nn: NeuralNetwork,
    game: BoardGame,
    opponent_evaluator,
    input_type: str = "board_flat",
    game_type: str = "puzzle8",
    shuffle_range: tuple[int, int] = DEFAULT_SHUFFLE_RANGE,
) -> dict[str, Any]:
    """
    Play an adversarial puzzle game: NN vs opponent resolve the same puzzle.

    1. Reset + shuffle the puzzle
    2. Both players get a copy of the same shuffled state
    3. Both solve independently
    4. Compare: tiles_in_place (desc), then moves (asc)

    Returns:
        Dict with 'result' ('win', 'loss', 'draw'), 'moves', 'record'
    """
    rng = random.Random()
    shuffle_moves = rng.randint(*shuffle_range)

    game.reset()
    game.shuffle(shuffle_moves, rng)

    # Identical copies for both players
    nn_game = game.copy()
    opp_game = game.copy()

    # NN solves its copy
    nn_eval = _get_nn_evaluator(nn, input_type)
    _solve_puzzle(nn_eval, nn_game)

    # Opponent solves its copy
    _solve_puzzle(opponent_evaluator, opp_game)

    # Comparison via game's compare()
    cmp = nn_game.compare(opp_game)
    if cmp > 0:
        result = "win"
    elif cmp < 0:
        result = "loss"
    else:
        result = "draw"

    record = {
        "shuffle_moves": shuffle_moves,
        "result": result,
        "opponent": type(opponent_evaluator).__name__,
    }

    return {"result": result, "moves": 0, "record": record}


def _play_single_game(
    nn: NeuralNetwork,
    game: BoardGame,
    opponent_evaluator,
    nn_player: int,
    input_type: str = "board_only",
    game_type: str = "tictactoe",
) -> dict[str, Any]:
    """
    Play a single game.

    Returns:
        Dict with 'result' ('win', 'loss', 'draw') and 'moves' count
    """
    game.reset()
    nn_eval = _get_nn_evaluator(nn, input_type)
    moves = 0

    while not game.is_terminal():
        if game.current_player == nn_player:
            move = nn_eval.get_move(game)
        else:
            move = opponent_evaluator.get_move(game)

        game.make_move(move)
        moves += 1

    winner = game.check_winner()
    if winner == nn_player:
        result = "win"
    elif winner == -nn_player:
        result = "loss"
    else:
        result = "draw"

    record = game.get_record()
    if record is not None:
        record["nn_player"] = nn_player
        record["opponent"] = type(opponent_evaluator).__name__

    return {"result": result, "moves": moves, "record": record}


def _evaluate_nn(
    nn: NeuralNetwork,
    game_type: str,
    opponent: str,
    num_games: int,
    opponent_mix: dict[str, float] | None = None,
    input_type: str = "board_only",
    record_games: bool = False,
    hof_opponents: list[list[float]] | None = None,
) -> EvalResult:
    """
    Evaluate a neural network by playing games.

    Unified flow for all game types (adversarial and puzzle).
    """
    try:
        game = _get_game(game_type)
    except ValueError as e:
        return EvalResult(success=False, error=str(e))

    is_puzzle = game.game_category == "puzzle"

    if record_games and not is_puzzle:
        game.enable_recording()

    # Track results
    results = {"win": 0, "loss": 0, "draw": 0}
    total_moves: dict[str, list[int]] = {"win": [], "loss": [], "draw": []}
    game_records: list[dict[str, Any]] = []

    # Build HoF evaluators if provided
    hof_evaluators = []
    if hof_opponents:
        for hof_genes in hof_opponents:
            hof_nn = NeuralNetwork.from_flat(
                flat=np.array(hof_genes),
                architecture=nn.architecture,
                activation=nn.activation,
                output_activation=nn.output_activation,
            )
            hof_evaluators.append(_get_nn_evaluator(hof_nn, input_type))

    # Get opponent evaluators
    if opponent_mix:
        opponents = {
            name: _get_evaluator(game_type, name)
            for name in opponent_mix
            if name != "hall_of_fame"
        }
    else:
        opponents = {opponent: _get_evaluator(game_type, opponent)}
        opponent_mix = {opponent: 1.0}

    for _ in range(num_games):
        # Choose opponent based on mix
        r = random.random()
        cumsum = 0.0
        chosen_opponent = opponent
        for opp_name, prob in opponent_mix.items():
            cumsum += prob
            if r < cumsum:
                chosen_opponent = opp_name
                break

        if chosen_opponent == "hall_of_fame" and hof_evaluators:
            opponent_eval = random.choice(hof_evaluators)
        elif chosen_opponent == "hall_of_fame":
            # Fallback: pas de HoF disponible → random
            opponent_eval = _get_evaluator(game_type, "random")
        else:
            opponent_eval = opponents[chosen_opponent]

        if is_puzzle:
            # Adversarial puzzle: both solve the same puzzle
            game_result = _play_puzzle_game(
                nn, game, opponent_eval, input_type, game_type,
            )
        else:
            # Adversarial board game: alternating turns
            nn_player = random.choice([1, -1])
            game_result = _play_single_game(
                nn, game, opponent_eval, nn_player, input_type, game_type,
            )

        result = game_result["result"]
        results[result] += 1
        total_moves[result].append(game_result["moves"])

        if record_games and game_result.get("record") is not None:
            game_records.append(game_result["record"])

    # Calculate metrics
    win_rate = results["win"] / num_games
    loss_rate = results["loss"] / num_games
    draw_rate = results["draw"] / num_games

    metrics = {
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "draw_rate": draw_rate,
        "wins": results["win"],
        "losses": results["loss"],
        "draws": results["draw"],
        "avg_moves_win": np.mean(total_moves["win"]) if total_moves["win"] else 0,
        "avg_moves_loss": np.mean(total_moves["loss"]) if total_moves["loss"] else 0,
        "avg_moves_draw": np.mean(total_moves["draw"]) if total_moves["draw"] else 0,
    }

    return EvalResult(
        success=True,
        metrics=metrics,
        games_played=num_games,
        game_records=game_records if record_games else None,
    )


def execute_eval_task(task: EvalTask) -> EvalResult:
    """
    Execute an evaluation task.

    This is the function passed to the process pool.
    Must be a top-level function (picklable).
    """
    try:
        # Reconstruct neural network from genes
        nn = NeuralNetwork.from_flat(
            flat=np.array(task.genes),
            architecture=task.architecture,
            activation=task.activation,
            output_activation=task.output_activation,
        )

        # Evaluate
        result = _evaluate_nn(
            nn=nn,
            game_type=task.game_type,
            opponent=task.opponent,
            num_games=task.num_games,
            opponent_mix=task.opponent_mix,
            input_type=task.input_type,
            record_games=task.record_games,
            hof_opponents=task.hof_opponents,
        )

        return result

    except Exception as e:
        return EvalResult(success=False, error=str(e))


def execute_supervised_task(task: SupervisedTask) -> SupervisedResult:
    """
    Execute a supervised training task.

    Top-level function (picklable for process pools).
    """
    try:
        nn = NeuralNetwork.from_flat(
            flat=np.array(task.genes),
            architecture=task.architecture,
            activation=task.activation,
            output_activation=task.output_activation,
        )
        game = _get_game(task.game_type)

        if task.method != "backprop":
            # Delegate to game's train_supervised for unknown methods
            modified_genes = game.train_supervised(
                nn, task.records, task.input_type, task.method, **task.training_params
            )
            return SupervisedResult(success=True, genes=modified_genes.tolist())

        # For backprop, call lower-level API to capture loss history
        from gengame_sdk.nn.training import TrainingConfig, train_backprop

        x_train, y = game.extract_training_data(task.records, task.input_type)
        config = TrainingConfig(
            **{
                k: v
                for k, v in task.training_params.items()
                if k in TrainingConfig.__dataclass_fields__
            }
        )
        nn_copy = nn.copy()
        result = train_backprop(nn_copy, x_train, y, config)

        return SupervisedResult(
            success=True,
            genes=nn_copy.to_flat().tolist(),
            losses=result.losses,
            final_loss=result.final_loss,
        )
    except Exception as e:
        return SupervisedResult(success=False, error=str(e))


def _play_puzzle_match_game(
    eval1,
    eval2,
    game: BoardGame,
    shuffle_range: tuple[int, int] = TOURNAMENT_SHUFFLE_RANGE,
) -> tuple[str, dict[str, Any]]:
    """
    Play a single puzzle match game between two NN evaluators.

    Both resolve the same shuffled puzzle, comparison by tiles_in_place then moves.

    Returns:
        (result_str, record) where result_str is 'p1_win', 'p2_win', or 'draw'
    """
    rng = random.Random()
    shuffle_moves = rng.randint(*shuffle_range)

    game.reset()
    game.shuffle(shuffle_moves, rng)

    # Identical copies
    game1 = game.copy()
    game2 = game.copy()

    # Both players solve
    _solve_puzzle(eval1, game1)
    _solve_puzzle(eval2, game2)

    # Comparison via game's compare()
    cmp = game1.compare(game2)
    if cmp > 0:
        result_str = "p1_win"
    elif cmp < 0:
        result_str = "p2_win"
    else:
        result_str = "draw"

    record = {
        "shuffle_moves": shuffle_moves,
    }

    return result_str, record


def execute_match_task(task: MatchTask) -> MatchResult:
    """
    Execute a match between two neural networks.

    Top-level function (picklable for process pools).
    Plays num_games games. For board games, alternates who starts first.
    For puzzles, both resolve the same shuffled puzzle.
    """
    try:
        # Seed for reproducibility
        random.seed(task.seed)
        np.random.seed(task.seed)

        # Reconstruct networks
        nn1 = NeuralNetwork.from_flat(
            flat=np.array(task.player1_genes),
            architecture=task.architecture,
            activation=task.activation,
            output_activation=task.output_activation,
        )
        nn2 = NeuralNetwork.from_flat(
            flat=np.array(task.player2_genes),
            architecture=task.player2_architecture or task.architecture,
            activation=task.player2_activation or task.activation,
            output_activation=task.player2_output_activation or task.output_activation,
        )

        p2_input_type = task.player2_input_type or task.input_type
        eval1 = _get_nn_evaluator(nn1, task.input_type)
        eval2 = _get_nn_evaluator(nn2, p2_input_type)

        game = _get_game(task.game_type)
        is_puzzle = game.game_category == "puzzle"

        p1_wins = 0
        p2_wins = 0
        draws = 0
        game_records = []

        for game_num in range(task.num_games):
            if is_puzzle:
                result_str, record = _play_puzzle_match_game(eval1, eval2, game)
                record["game_num"] = game_num
            else:
                game.reset()

                # Alternate who starts: even games p1=1, odd games p1=-1
                if game_num % 2 == 0:
                    p1_symbol = 1
                else:
                    p1_symbol = -1

                while not game.is_terminal():
                    if game.current_player == p1_symbol:
                        move = eval1.get_move(game)
                    else:
                        move = eval2.get_move(game)
                    game.make_move(move)

                winner = game.check_winner()
                if winner == p1_symbol:
                    result_str = "p1_win"
                elif winner == -p1_symbol:
                    result_str = "p2_win"
                else:
                    result_str = "draw"

                record = {
                    "game_num": game_num,
                    "p1_symbol": p1_symbol,
                    "result": result_str,
                }

            if result_str == "p1_win":
                p1_wins += 1
            elif result_str == "p2_win":
                p2_wins += 1
            else:
                draws += 1

            record["result"] = result_str
            game_records.append(record)

        return MatchResult(
            success=True,
            player1_wins=p1_wins,
            player2_wins=p2_wins,
            draws=draws,
            game_records=game_records,
        )

    except Exception as e:
        return MatchResult(success=False, error=str(e))
