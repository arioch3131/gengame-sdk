# GenGame SDK Architecture

## Overview

The GenGame SDK provides the foundational abstractions and services for building GenGame-compatible game plugins. It is designed to be installed as a dependency by both the core GenGame platform and individual game implementations.

## Components

### Core Abstractions

| Module | Description |
|--------|-------------|
| `game.py` | `BoardGame` ABC — defines the interface all games must implement |
| `evaluator.py` | `Evaluator` ABC — defines the interface for AI opponents |
| `plugin.py` | `GamePlugin` dataclass + `discover_games()` via entry points |

### Neural Network (`nn/`)

Pure NumPy feedforward neural network, serializable to a 1D gene array for genetic algorithm compatibility.

| File | Description |
|------|-------------|
| `network.py` | `NeuralNetwork` dataclass, `create_network()` factory, activations, initializations |
| `training.py` | `train_backprop()` — forward with cache, cross-entropy loss, backward pass, SGD |

### Game API Server (`server/`)

Generic FastAPI service that serves any installed game plugin via HTTP. Included in the SDK because every game Docker image needs it.

| File | Description |
|------|-------------|
| `app.py` | `create_app()` FastAPI factory with lifespan management |
| `routes.py` | HTTP endpoints: evaluate, match, supervised, play, info |
| `schemas.py` | Pydantic request/response models |
| `service.py` | Game execution logic (evaluate, match, supervised training) |
| `tasks.py` | Task/Result dataclasses for eval, match, supervised, batch |
| `play.py` | `SessionManager` for interactive play sessions with TTL |

## Plugin Discovery

Games register via Python entry points (`gengame.games` group). Each game provides a `register()` function returning a `GamePlugin`-compatible dict with:
- `game_class`: BoardGame subclass
- `evaluators`: Dict of evaluator factories
- `get_evaluator`: Factory function
- `nn_evaluator_class`: NeuralNetworkEvaluator class
- `player_symbols`: Display symbols
- `move_prompt`: Interactive play prompt

## Data Flow

```
Client request -> FastAPI routes -> service.py (execute_*_task)
                                        |
                                NeuralNetwork.from_flat(genes)
                                        |
                                get_game_plugin(game_type)
                                        |
                                Play games / train / match
                                        |
                                Return Result dataclass
```
