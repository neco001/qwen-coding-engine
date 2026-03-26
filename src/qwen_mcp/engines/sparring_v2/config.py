"""
Sparring Engine v2 - Configuration Constants

This module provides timeout and default model configuration for the sparring engine.
"""

# =============================================================================
# Timeout Configuration (reduced to avoid MCP 300s limit)
# =============================================================================

TIMEOUTS = {
    "flash_analyst": 90.0,      # Reduced from 300s
    "flash_drafter": 90.0,       # Reduced from 300s
    "discovery": 60.0,           # Keep as is (already low)
    "red_cell": 90.0,            # Reduced from 300s
    "blue_cell": 90.0,           # Reduced from 300s
    "white_cell": 90.0,          # Reduced from 300s
}

# Default models for each cell role
DEFAULT_MODELS = {
    "red_model": "glm-5",
    "blue_model": "qwen3.5-plus",
    "white_model": "qwen3.5-plus",
}