"""
Game API server — FastAPI service for game evaluation via HTTP.
"""

from .app import create_app

__all__ = ["create_app"]
