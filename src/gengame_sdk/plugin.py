"""
Plugin discovery for GenGame games via entry points.
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass, field

from .evaluator import Evaluator
from .game import BoardGame

if sys.version_info >= (3, 12):
    from importlib.metadata import entry_points
else:
    from importlib.metadata import entry_points


@dataclass
class GamePlugin:
    """Description of an installed game."""

    game_class: type[BoardGame]
    evaluators: dict[str, Callable[[], Evaluator]]
    get_evaluator: Callable[[str], Evaluator]
    player_symbols: dict[int, str] = field(default_factory=dict)
    move_prompt: str = ""


# Internal cache
_cache: dict[str, GamePlugin] | None = None


def discover_games() -> dict[str, GamePlugin]:
    """Discover installed games via entry points group='gengame.games'."""
    global _cache  # noqa: PLW0603
    if _cache is not None:
        return _cache

    plugins: dict[str, GamePlugin] = {}
    eps = entry_points(group="gengame.games")
    for ep in eps:
        register_fn = ep.load()
        info = register_fn()
        plugins[ep.name] = GamePlugin(**info)

    _cache = plugins
    return plugins


def get_game_plugin(name: str) -> GamePlugin:
    """Return a plugin by name, raises ValueError if unknown."""
    plugins = discover_games()
    if name not in plugins:
        available = list(plugins.keys())
        raise ValueError(f"Unknown game type: {name}. Available: {available}")
    return plugins[name]


def clear_cache() -> None:
    """Clear the cache (for tests)."""
    global _cache  # noqa: PLW0603
    _cache = None
