"""Tests for plugin discovery system."""

import pytest

from gengame_sdk.plugin import GamePlugin, clear_cache, discover_games, get_game_plugin


class TestDiscoverGames:
    """Test discover_games."""

    def test_returns_dict(self):
        clear_cache()
        games = discover_games()
        assert isinstance(games, dict)

    def test_caching(self):
        clear_cache()
        games1 = discover_games()
        games2 = discover_games()
        assert games1 is games2


class TestClearCache:
    """Test clear_cache."""

    def test_clears_cache(self):
        clear_cache()
        games1 = discover_games()
        clear_cache()
        games2 = discover_games()
        # After clearing, a new dict is created
        assert games1 is not games2


class TestGetGamePlugin:
    """Test get_game_plugin."""

    def test_unknown_game_raises(self):
        clear_cache()
        with pytest.raises(ValueError, match="Unknown game type"):
            get_game_plugin("nonexistent_game_xyz")


class TestGamePlugin:
    """Test GamePlugin dataclass."""

    def test_defaults(self):
        from gengame_sdk.game import BoardGame

        class DummyGame(BoardGame):
            @property
            def current_player(self): return 1
            @property
            def board_size(self): return 9
            def reset(self): pass
            def get_valid_moves(self): return [0]
            def make_move(self, move): return True
            def check_winner(self): return 0
            def is_terminal(self): return False
            def get_board_state(self): return [0] * 9
            def get_board_for_player(self, p): return [0.0] * 9
            def get_input(self, p, t): return [0.0] * 9
            def copy(self): return self
            def display(self): return ""

        plugin = GamePlugin(
            game_class=DummyGame,
            evaluators={},
            get_evaluator=lambda n: None,
        )
        assert plugin.player_symbols == {}
        assert plugin.move_prompt == ""
