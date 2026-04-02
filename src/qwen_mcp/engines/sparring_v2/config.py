"""
Sparring Engine v2 - Configuration Constants

This module provides timeout and default model configuration for the sparring engine.

Sparring Levels:
- sparring1 (flash): 2 steps (analyst→drafter), 180s total timeout
- sparring2 (normal): 4 steps in one call (full), 180s total timeout
- sparring3 (pro): 4 steps separately (step-by-step), 100s per step

Note: MODE_ALIASES and DEFAULT_SPARRING_MODE are defined in tools.py to avoid circular imports.
"""

# =============================================================================
# Timeout Configuration
# =============================================================================
# sparring1 (flash): 90s + 90s = 180s total
# sparring2 (normal/full): Uses step timeouts with progress reporting
# sparring3 (pro): 100s per step for step-by-step execution

TIMEOUTS = {
    "flash_analyst": 90.0,       # sparring1: 180s total for 2 steps
    "flash_drafter": 90.0,       # sparring1: 180s total for 2 steps
    "discovery": 100.0,          # sparring3: step-by-step, 100s per step
    "red_cell": 100.0,           # sparring3: step-by-step, 100s per step
    "blue_cell": 100.0,          # sparring3: step-by-step, 100s per step
    "white_cell": 100.0,         # sparring3: step-by-step, 100s per step
}

# Note: WORD_LIMITS are defined in prompts/sparring.py

# Default models for each cell role
DEFAULT_MODELS = {
    "red_model": "glm-5",
    "blue_model": "qwen3.5-plus",
    "white_model": "qwen3.5-plus",
}