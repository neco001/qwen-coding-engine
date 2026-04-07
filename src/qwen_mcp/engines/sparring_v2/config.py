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

# =============================================================================
# MAX TOKENS CONFIGURATION
# =============================================================================
# Controls output length for each sparring mode/step
# Format: tokens per step
#
# sparring1 (flash): 1024 tokens total (fast 2-step analysis)
# sparring2 (full): 3584 tokens total (512 + 1024*3, balanced for 180s timeout)
# sparring3 (pro): 12800 tokens total (512 + 4096*3, deep analysis)

MAX_TOKENS_CONFIG = {
    "flash": {
        "analyst": 1024,
        "drafter": 1024,
    },
    "full": {
        "discovery": 512,
        "red": 1024,
        "blue": 1024,
        "white": 1024,
    },
    "pro": {
        "discovery": 512,
        "red": 4096,
        "blue": 4096,
        "white": 4096,
    },
}


# =============================================================================
# MAX THINKING TOKENS CONFIGURATION
# =============================================================================
# Controls thinking tokens (enable_thinking output) for each sparring mode.
# Thinking tokens are used for the model's internal reasoning process.
#
# sparring1 (flash): 512 thinking tokens per step (fast 2-step analysis)
# sparring2 (full): 256 thinking tokens per step (reduced to fix 180s timeout)
# sparring3 (pro): 1024 thinking tokens per step (deep analysis, 100s per step)

MAX_THINKING_TOKENS_CONFIG = {
    "sparring1": 512,
    "sparring2": 256,
    "sparring3": 1024,
}


def get_thinking_tokens_for_mode(mode: str) -> int:
    """
    Get max_thinking_tokens for a specific sparring mode.
    
    Args:
        mode: One of 'sparring1', 'sparring2', 'sparring3'
    
    Returns:
        max_thinking_tokens value (default: 256 if mode not found)
    """
    return MAX_THINKING_TOKENS_CONFIG.get(mode, 256)


def get_max_tokens_for_step(mode: str, step: str) -> int:
    """
    Get max_tokens for a specific sparring mode and step.
    
    Args:
        mode: One of 'flash', 'full', 'pro'
        step: One of 'analyst', 'drafter', 'discovery', 'red', 'blue', 'white'
    
    Returns:
        max_tokens value (default: 2048 if not found)
    """
    return MAX_TOKENS_CONFIG.get(mode, {}).get(step, 2048)


# Note: WORD_LIMITS are defined in prompts/sparring.py

# Default models for each cell role
DEFAULT_MODELS = {
    "red_model": "glm-5",
    "blue_model": "qwen3.5-plus",
    "white_model": "qwen3.5-plus",
}