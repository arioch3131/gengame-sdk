"""
Interactive play session management via the API.

Stores sessions in memory with TTL and periodic cleanup.
"""

import asyncio
import random
import time
import uuid

import numpy as np

from gengame_sdk.nn import NeuralNetwork
from gengame_sdk.nn_evaluator import NeuralNetworkEvaluator
from gengame_sdk.plugin import get_game_plugin

# Session TTL in seconds (30 minutes)
SESSION_TTL = 30 * 60


class PlaySession:
    """Interactive play session: human vs AI."""

    def __init__(self, game, nn_evaluator, human_player: int, ai_player: int):
        self.game = game
        self.nn_evaluator = nn_evaluator
        self.human_player = human_player
        self.ai_player = ai_player
        self.last_access = time.time()

    def touch(self):
        self.last_access = time.time()

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.last_access) > SESSION_TTL


class SessionManager:
    """Play session manager with automatic cleanup."""

    def __init__(self):
        self._sessions: dict[str, PlaySession] = {}
        self._cleanup_task: asyncio.Task | None = None

    def start_cleanup(self) -> None:
        """Start periodic cleanup of expired sessions."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    def stop_cleanup(self) -> None:
        """Stop periodic cleanup."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        """Cleanup loop every 5 minutes."""
        while True:
            await asyncio.sleep(300)
            self._cleanup_expired()

    def _cleanup_expired(self) -> None:
        """Remove expired sessions."""
        expired = [sid for sid, s in self._sessions.items() if s.is_expired]
        for sid in expired:
            del self._sessions[sid]

    def create_session(
        self,
        game_type: str,
        genes: list[float],
        architecture: list[int],
        activation: str,
        output_activation: str,
        input_type: str,
    ) -> tuple[str, PlaySession]:
        """Create a new play session."""
        nn = NeuralNetwork.from_flat(
            flat=np.array(genes),
            architecture=architecture,
            activation=activation,
            output_activation=output_activation,
        )

        plugin = get_game_plugin(game_type)
        game = plugin.game_class()
        game.reset()

        nn_evaluator = NeuralNetworkEvaluator(nn, input_type=input_type)

        human_player = random.choice([1, -1])
        ai_player = -human_player

        session = PlaySession(game, nn_evaluator, human_player, ai_player)
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = session

        return session_id, session

    def get_session(self, session_id: str) -> PlaySession | None:
        """Get a session by ID."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired:
            del self._sessions[session_id]
            return None
        session.touch()
        return session

    def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        self._sessions.pop(session_id, None)
