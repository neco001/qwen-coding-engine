"""
Token Validator - Pre-flight token budget validation.

This module provides validation for token budgets before API calls
to prevent HTTP 400 errors from DashScope API.

Key Features:
- validate_token_budget(): Check if input exceeds limit
- TokenBudgetExceeded: Custom exception for budget violations
- Heuristic token counting (no external dependencies)
"""

import logging
from typing import Optional

logger = logging.getLogger("qwen_mcp.token_validator")


class TokenBudgetExceeded(Exception):
    """
    Raised when token budget exceeds the configured limit.
    
    Attributes:
        estimated_tokens: Estimated token count
        limit: Configured token limit
        message: Human-readable error message
    """
    
    def __init__(self, estimated_tokens: int, limit: int, message: Optional[str] = None):
        self.estimated_tokens = estimated_tokens
        self.limit = limit
        self.message = message or f"Token budget exceeded: {estimated_tokens} tokens estimated (limit: {limit})"
        super().__init__(self.message)


def validate_token_budget(
    input_text: str,
    limit: int = 60000,
    context_text: str = ""
) -> int:
    """
    Validate that input + context token count is within budget.
    
    Uses heuristic estimation (~4 characters per token for English/code).
    This is a pre-flight check BEFORE calling TokenScout or API.
    
    Args:
        input_text: The user prompt/input text
        limit: Maximum allowed tokens (default: 60000, buffer below 65536 API limit)
        context_text: Additional context (file contents, etc.)
    
    Returns:
        Estimated token count
    
    Raises:
        TokenBudgetExceeded: If estimated tokens exceed limit
    
    Example:
        >>> validate_token_budget("Write a function...", limit=60000)
        1500
        
        >>> validate_token_budget(huge_text, limit=60000)
        TokenBudgetExceeded: Token budget exceeded: 75000 tokens estimated (limit: 60000)
    """
    # Heuristic: ~4 characters per token for English/code
    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(text) // 4
    
    input_tokens = count_tokens(input_text)
    context_tokens = count_tokens(context_text) if context_text else 0
    total_estimated = input_tokens + context_tokens
    
    if total_estimated > limit:
        logger.warning(
            f"Token budget exceeded: {total_estimated} tokens (input: {input_tokens}, "
            f"context: {context_tokens}) > limit {limit}"
        )
        raise TokenBudgetExceeded(
            estimated_tokens=total_estimated,
            limit=limit,
            message=f"Input too large: ~{total_estimated} tokens estimated (limit: {limit} tokens). "
                    f"Consider splitting into smaller tasks or using segmentation."
        )
    
    logger.debug(f"Token budget OK: {total_estimated} tokens (limit: {limit})")
    return total_estimated


def estimate_tokens_heuristic(text: str) -> int:
    """
    Quick token estimation using character count heuristic.
    
    Rule of thumb: ~4 characters per token for English/code.
    This matches TokenScout's internal heuristic for consistency.
    
    Args:
        text: Text to estimate
    
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return len(text) // 4
