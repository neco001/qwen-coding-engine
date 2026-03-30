"""
Sparring Engine v2 - Configuration Constants

This module provides timeout and default model configuration for the sparring engine.
"""

# =============================================================================
# Timeout Configuration (reduced to avoid MCP 300s limit)
# =============================================================================

TIMEOUTS = {
    "flash_analyst": 30.0,       # Deep thinking models need more time
    "flash_drafter": 30.0,       # Flash total: ~60s (acceptable for single call)
    "discovery": 20.0,           # JSON extraction with thinking buffer
    "red_cell": 45.0,            # Deep analysis with full thinking budget
    "blue_cell": 45.0,           # Strategic defense needs space
    "white_cell": 45.0,          # Synthesis requires complete context
}

# Default models for each cell role
DEFAULT_MODELS = {
    "red_model": "glm-5",
    "blue_model": "qwen3.5-plus",
    "white_model": "qwen3.5-plus",
}