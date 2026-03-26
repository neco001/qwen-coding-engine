"""
Sparring Engine v2 - Public API

This module provides backward-compatible imports for the sparring engine.
All public symbols are re-exported from their respective modules.
"""

from qwen_mcp.engines.sparring_v2.config import TIMEOUTS, DEFAULT_MODELS
from qwen_mcp.engines.sparring_v2.models import SparringResponse
from qwen_mcp.engines.sparring_v2.engine import SparringEngineV2

__all__ = [
    "SparringEngineV2",
    "SparringResponse",
    "TIMEOUTS",
    "DEFAULT_MODELS",
]