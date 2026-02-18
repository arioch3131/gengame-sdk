"""Tests for BoardGame abstract base class."""

import pytest

from gengame_sdk.game import BoardGame


class TestBoardGameABC:
    """Test that BoardGame enforces abstract methods."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BoardGame()

    def test_minimal_implementation(self):
        class MinimalGame(BoardGame):
            @property
            def current_player(self): return 1
            @property
            def board_size(self): return 4
            def reset(self): pass
            def get_valid_moves(self): return [0, 1]
            def make_move(self, move): return True
            def check_winner(self): return 0
            def is_terminal(self): return False
            def get_board_state(self): return [0, 0, 0, 0]
            def get_board_for_player(self, p): return [0.0] * 4
            def get_input(self, p, t): return [0.0] * 4
            def copy(self): return MinimalGame()
            def display(self): return "board"

        game = MinimalGame()
        assert game.current_player == 1
        assert game.board_size == 4
        assert game.output_size == 4  # defaults to board_size
        assert game.input_size == 4  # defaults to board_size
        assert game.game_category == "adversarial"

    def test_missing_abstract_method(self):
        with pytest.raises(TypeError):
            class IncompleteGame(BoardGame):
                @property
                def current_player(self): return 1
            IncompleteGame()


class TestBoardGameDefaults:
    """Test default method implementations."""

    def _make_game(self, winner=0, terminal=False):
        class TestGame(BoardGame):
            @property
            def current_player(self): return 1
            @property
            def board_size(self): return 9
            def reset(self): pass
            def get_valid_moves(self): return [0]
            def make_move(self, move): return True
            def check_winner(self): return winner
            def is_terminal(self): return terminal
            def get_board_state(self): return [0] * 9
            def get_board_for_player(self, p): return [0.0] * 9
            def get_input(self, p, t): return [0.0] * 9
            def copy(self): return self
            def display(self): return ""
        return TestGame()

    def test_get_result_not_terminal(self):
        game = self._make_game(winner=0, terminal=False)
        assert game.get_result() is None

    def test_get_result_player1_wins(self):
        game = self._make_game(winner=1, terminal=True)
        assert game.get_result() == "player1"

    def test_get_result_player2_wins(self):
        game = self._make_game(winner=-1, terminal=True)
        assert game.get_result() == "player2"

    def test_get_result_draw(self):
        game = self._make_game(winner=0, terminal=True)
        assert game.get_result() == "draw"

    def test_recording_default(self):
        game = self._make_game()
        assert game.is_recording is False
        game.enable_recording()
        assert game.is_recording is True
        game.disable_recording()
        assert game.is_recording is False

    def test_get_record_default(self):
        game = self._make_game()
        assert game.get_record() is None

    def test_get_training_methods_default(self):
        game = self._make_game()
        assert game.get_training_methods() == []

    def test_extract_training_data_not_implemented(self):
        game = self._make_game()
        with pytest.raises(NotImplementedError):
            game.extract_training_data([], "board_only")

    def test_available_input_types_default(self):
        game = self._make_game()
        assert game.get_available_input_types() == ["default"]
