"""
Sparring Engine v2 - Configuration Constants

This module provides timeout and default model configuration for the sparring engine.

Sparring Levels:
- sparring1 (flash): 2 steps (analyst→drafter), 180s total timeout
- sparring2 (normal): 4 steps in one call (full), 180s total timeout
- sparring3 (pro): 4 steps separately (step-by-step), 100s per step
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

# Default models for each cell role
DEFAULT_MODELS = {
    "red_model": "glm-5",
    "blue_model": "qwen3.5-plus",
    "white_model": "qwen3.5-plus",
}

# =============================================================================
# Mode Aliases (sparring1/2/3 → internal modes)
# =============================================================================

MODE_ALIASES = {
    "sparring1": "flash",    # Quick 2-step analysis
    "sparring2": "full",     # Full session in one call (DEFAULT)
    "sparring3": "pro",      # Step-by-step with checkpointing
    # Short aliases
    "nor": "full",           # "normal" shortcut
    # Legacy aliases (passthrough)
    "flash": "flash",
    "full": "full",
    "pro": "pro",
    "discovery": "discovery",
    "red": "red",
    "blue": "blue",
    "white": "white",
}

# Default sparring level
DEFAULT_SPARRING_MODE = "sparring2"  # normal/full mode