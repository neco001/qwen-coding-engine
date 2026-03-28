"""
Sparring Engine v2 - Configuration Constants

This module provides timeout and default model configuration for the sparring engine.
"""

# =============================================================================
# Timeout Configuration (reduced to avoid MCP 300s limit)
# =============================================================================

TIMEOUTS = {
    "flash_analyst": 25.0,       # Increased to allow for 'Deep Thinking' (Heartbeat will prevent timeout)
    "flash_drafter": 25.0,
    "discovery": 30.0,
    "red_cell": 45.0,            # High complexity audit
    "blue_cell": 45.0,
    "white_cell": 45.0,
}

# Default models for each cell role
DEFAULT_MODELS = {
    "red_model": "glm-5",
    "blue_model": "qwen3.5-plus",
    "white_model": "qwen3.5-plus",
}