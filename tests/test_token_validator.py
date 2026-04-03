"""Token Validator Tests for boundary conditions (59K, 60K, 61K tokens)."""

import pytest
from qwen_mcp.engines.token_validator import validate_token_budget, TokenBudgetExceeded, estimate_tokens_heuristic


class TestTokenValidator:
    """Tests for token budget validation boundary conditions."""

    def test_under_budget_passes(self):
        """Test that inputs under 60K tokens pass validation."""
        # Generate text that estimates to ~59K tokens (59000 * 4 = 236000 chars)
        text = "x" * 236000
        result = validate_token_budget(text, limit=60000)
        assert result <= 60000

    def test_at_budget_boundary_passes(self):
        """Test that exactly 60K tokens passes validation."""
        # Generate text that estimates to ~60K tokens (60000 * 4 = 240000 chars)
        text = "x" * 240000
        result = validate_token_budget(text, limit=60000)
        assert result <= 60000

    def test_over_budget_raises_exception(self):
        """Test that inputs over 60K tokens raise TokenBudgetExceeded."""
        # Generate text that estimates to ~61K tokens (61000 * 4 = 244000 chars)
        text = "x" * 244000
        with pytest.raises(TokenBudgetExceeded) as exc_info:
            validate_token_budget(text, limit=60000)
        
        # Verify exception contains useful information
        assert "61000" in str(exc_info.value) or "tokens" in str(exc_info.value).lower()

    def test_custom_limit(self):
        """Test that custom limits work correctly."""
        # Test with a smaller limit
        text = "x" * 4000  # ~1K tokens
        result = validate_token_budget(text, limit=5000)
        assert result <= 5000

    def test_context_tokens_included(self):
        """Test that context tokens are included in estimation."""
        input_text = "x" * 100000  # ~25K tokens
        context_text = "y" * 140000  # ~35K tokens
        # Total should be ~60K tokens
        result = validate_token_budget(input_text, limit=60000, context_text=context_text)
        assert result <= 60000

    def test_empty_input_passes(self):
        """Test that empty input passes validation."""
        result = validate_token_budget("", limit=60000)
        assert result == 0

    def test_estimate_tokens_heuristic(self):
        """Test the token estimation heuristic."""
        # Rule of thumb: ~4 characters per token
        assert estimate_tokens_heuristic("xxxx") == 1
        assert estimate_tokens_heuristic("xxxxxxxx") == 2
        assert estimate_tokens_heuristic("") == 0
        assert estimate_tokens_heuristic("x") == 0  # Less than 4 chars = 0 tokens
