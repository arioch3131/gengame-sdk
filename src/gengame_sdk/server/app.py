"""
FastAPI application for the Game API.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .routes import _get_game_info, router, session_manager, set_executor, shutdown_executor
from .schemas import GameInfoResponse


def create_app(
    game_type: str | None = None,
    workers: int | None = None,
) -> FastAPI:
    """
    Create the FastAPI application for the Game API.

    Args:
        game_type: Game type to serve. If None, serves all available games.
        workers: Number of workers for the ProcessPoolExecutor.
    """
    if workers is None:
        workers = os.cpu_count() or 4

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        executor = ProcessPoolExecutor(max_workers=workers)
        set_executor(executor)
        session_manager.start_cleanup()
        yield
        # Shutdown
        session_manager.stop_cleanup()
        shutdown_executor()

    app = FastAPI(
        title="GenGame Game API",
        description="HTTP API for GenGame game evaluation",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(router)

    # Store game_type for /info endpoint
    app.state.game_type = game_type

    @app.get("/info", response_model=list[GameInfoResponse] | GameInfoResponse)
    async def info():
        """Return information about the served game(s)."""
        if app.state.game_type:
            return _get_game_info(app.state.game_type)

        # Multi-game mode: return all available games
        from gengame_sdk.plugin import discover_games

        available_games = list(discover_games().keys())
        return [_get_game_info(gt) for gt in available_games]

    return app
