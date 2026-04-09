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
    "discovery": 120.0,          # sparring3: step-by-step, 120s per step
    "red_cell": 180.0,           # sparring3: step-by-step, 180s per step (4096 tokens)
    "blue_cell": 180.0,          # sparring3: step-by-step, 180s per step (4096 tokens)
    "white_cell": 360.0,         # sparring3: step-by-step, 360s per step (8192 tokens + synthesis + artifact)
}

# =============================================================================
# STAGE WEIGHTS CONFIGURATION
# =============================================================================
# Default stage weights for BudgetManager in stage-based executors
# Weights are normalized to sum to 1.0 for each mode
#
# pro (sparring3): 4 stages, 225s total budget
# full (sparring2): 4 stages, 225s total budget (includes regeneration)
# flash (sparring1): 2 stages, 60s total budget

STAGE_WEIGHTS = {
    "pro": {
        "discovery": 0.15,  # 33.75s
        "red": 0.28,        # 63s
        "blue": 0.28,       # 63s
        "white": 0.29,      # 65.25s
    },
    "full": {
        "discovery": 0.15,
        "red": 0.28,
        "blue": 0.28,
        "white": 0.29,  # includes regeneration budget
    },
    "flash": {
        "analyst": 0.45,   # 27s
        "drafter": 0.55,   # 33s
    },
}

# =============================================================================
# BUDGET CONFIGURATION
# =============================================================================
# Total timeout budgets for stage-based executors
# These budgets are managed by BudgetManager with dynamic allocation

BUDGET_CONFIG = {
    "pro": 900,      # 900 seconds for 4-stage execution (120+180+180+360 = 840s + buffer)
    "full": 225,     # 225 seconds (includes regeneration loop budget)
    "flash": 60,     # 60 seconds for fast 2-step analysis
}

# =============================================================================
# CIRCUIT BREAKER CONFIGURATION
# =============================================================================
# Circuit breaker settings for failure recovery in stage execution

CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": 3,   # Open circuit after 3 consecutive failures
    "recovery_timeout": 60,   # Wait 60 seconds before attempting recovery
}

# =============================================================================
# EPHEMERAL TTL CONFIGURATION
# =============================================================================
# TTL (Time-To-Live) for ephemeral checkpoints in flash mode
# Flash mode checkpoints expire after this duration

EPHEMERAL_TTL = 300  # 300 seconds (5 minutes) for flash mode

# =============================================================================
# MAX TOKENS CONFIGURATION
# =============================================================================
# Controls max_tokens for each sparring mode and step.
# max_tokens = thinking_budget + content_tokens (they share the same limit)
#
# Content limits for white cell:
# - sparring1/sparring2: 1024 content tokens
# - sparring3: 2048 content tokens

MAX_TOKENS_CONFIG = {
    "flash": {
        "analyst": 0,  # unlimited - controlled by thinking_budget
        "drafter": 0,  # unlimited - controlled by thinking_budget
    },
    "full": {
        "discovery": 0,  # unlimited - controlled by thinking_budget
        "red": 0,  # unlimited - controlled by thinking_budget
        "blue": 0,  # unlimited - controlled by thinking_budget
        "white": 0,  # unlimited - controlled by thinking_budget
    },
    "pro": {
        "discovery": 0,  # unlimited - controlled by thinking_budget
        "red": 0,  # unlimited - controlled by thinking_budget
        "blue": 0,  # unlimited - controlled by thinking_budget
        "white": 0,  # unlimited - controlled by thinking_budget
    },
}


# =============================================================================
# MAX THINKING TOKENS CONFIGURATION
# =============================================================================
# Controls thinking tokens (enable_thinking output) for each sparring mode and step.
# Thinking tokens are used for the model's internal reasoning process.
#
# sparring1 (flash): 1024 thinking tokens for all steps
# sparring2 (full): 1024 for red/blue, 2048 for white
# sparring3 (pro): 2048 for red/blue, 4096 for white

MAX_THINKING_TOKENS_CONFIG = {
    "sparring1": {
        "discovery": 1024,
        "red": 1024,
        "blue": 1024,
        "white": 1024,
    },
    "sparring2": {
        "discovery": 1024,
        "red": 1024,
        "blue": 1024,
        "white": 2048,
    },
    "sparring3": {
        "discovery": 2048,
        "red": 2048,
        "blue": 2048,
        "white": 4096,
    },
}


def get_thinking_tokens_for_mode(mode: str, step: str = "white") -> int:
    """
    Get max_thinking_tokens for a specific sparring mode and step.
    
    Args:
        mode: One of 'sparring1', 'sparring2', 'sparring3'
        step: One of 'discovery', 'red', 'blue', 'white' (default: 'white')
    
    Returns:
        max_thinking_tokens value (default: 1024 if mode/step not found)
    """
    return MAX_THINKING_TOKENS_CONFIG.get(mode, {}).get(step, 1024)


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