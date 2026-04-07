"""
Sparring Engine v2 - Mode Executors

This package contains mode-specific execution logic for the sparring engine.
Each module implements the ModeExecutor interface.
"""

from qwen_mcp.engines.sparring_v2.modes.flash import FlashExecutor
from qwen_mcp.engines.sparring_v2.modes.discovery import DiscoveryExecutor
from qwen_mcp.engines.sparring_v2.modes.red_cell import RedCellExecutor
from qwen_mcp.engines.sparring_v2.modes.blue_cell import BlueCellExecutor
from qwen_mcp.engines.sparring_v2.modes.white_cell import WhiteCellExecutor
from qwen_mcp.engines.sparring_v2.modes.full import FullExecutor

__all__ = [
    "FlashExecutor",
    "DiscoveryExecutor",
    "RedCellExecutor",
    "BlueCellExecutor",
    "WhiteCellExecutor",
    "FullExecutor",
]
