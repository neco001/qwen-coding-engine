"""
Sparring Engine v2 - Configuration Constants

This module provides timeout and default model configuration for the sparring engine.
"""

# =============================================================================
# Timeout Configuration (reduced to avoid MCP 300s limit)
# =============================================================================

TIMEOUTS = {
    "flash_analyst": 60.0,       # Reduced for MCP 300s limit
    "flash_drafter": 60.0,       # Reduced for MCP 300s limit
    "discovery": 45.0,           # Reduced for full mode
    "red_cell": 60.0,            # Reduced for full mode
    "blue_cell": 60.0,           # Reduced for full mode
    "white_cell": 60.0,          # Reduced for full mode (max 2 loops = 120s)
}

# Default models for each cell role
DEFAULT_MODELS = {
    "red_model": "glm-5",
    "blue_model": "qwen3.5-plus",
    "white_model": "qwen3.5-plus",
}