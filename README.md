# GenGame SDK

SDK for building GenGame-compatible games. Provides the core abstractions (BoardGame, Evaluator, NeuralNetwork) and a generic Game API server that any game plugin can use.

## Installation

```bash
pip install gengame-sdk
```

For development:

```bash
pip install -e ".[dev]"
```

## SDK Usage

### Implementing a game

Create a new game by subclassing `BoardGame` and `Evaluator`:

```python
from gengame_sdk import BoardGame, Evaluator

class MyGame(BoardGame):
    # Implement all abstract methods: reset(), make_move(), check_winner(), etc.
    ...

class MyEvaluator(Evaluator):
    def get_move(self, game: MyGame) -> int:
        ...
```

### Plugin registration

Register your game as a GenGame plugin in `pyproject.toml`:

```toml
[project.entry-points."gengame.games"]
mygame = "my_package:register"
```

And define the `register()` function:

```python
def register():
    return {
        "game_class": MyGame,
        "evaluators": {"random": RandomEvaluator},
        "get_evaluator": get_evaluator,
        "nn_evaluator_class": NeuralNetworkEvaluator,
        "player_symbols": {1: "X", -1: "O"},
        "move_prompt": "Enter move:",
    }
```

### Neural Network

Pure NumPy feedforward network with GA-compatible serialization:

```python
from gengame_sdk.nn import create_network, NeuralNetwork

nn = create_network([9, 32, 9], activation="tanh", output_activation="softmax", seed=42)
output = nn.forward(input_array)

# Serialize to/from flat gene array
genes = nn.to_flat()
nn2 = NeuralNetwork.from_flat(genes, [9, 32, 9])
```

## Game API Server

The SDK includes a generic FastAPI server that serves any installed game plugin:

```bash
# Serve a specific game
python -m uvicorn "gengame_sdk.server.app:create_app" --factory

# Or use from code
from gengame_sdk.server import create_app
app = create_app(game_type="tictactoe", workers=4)
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/evaluate` | POST | Evaluate a neural network by playing games |
| `/evaluate/batch` | POST | Batch evaluation |
| `/match` | POST | Play a match between two neural networks |
| `/supervised` | POST | Supervised training on game records |
| `/play/start` | POST | Start an interactive play session |
| `/play/move` | POST | Play a move in an interactive session |
| `/info` | GET | Get game information |

## Running Tests

```bash
pytest
```

## License

MIT
