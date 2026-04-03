"""
Test suite for Sparring Mode Aliases and Timeout Configuration.

Tests verify:
1. Mode aliases (sparring1/2/3 → flash/full/pro)
2. Timeout configuration (180s for sparring1/2, 100s for sparring3)
3. Default mode is sparring2 (normal)
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from qwen_mcp.engines.sparring_v2.config import TIMEOUTS, DEFAULT_MODELS
from qwen_mcp.tools import resolve_sparring_mode


class TestSparringModeAliases:
    """Test mode alias resolution."""
    
    def test_sparring1_alias_to_flash(self):
        """sparring1 should resolve to flash."""
        result = resolve_sparring_mode("sparring1")
        assert result == "flash", f"Expected 'flash', got '{result}'"
    
    def test_sparring2_alias_to_full(self):
        """sparring2 should resolve to full."""
        result = resolve_sparring_mode("sparring2")
        assert result == "full", f"Expected 'full', got '{result}'"
    
    def test_sparring3_alias_to_pro(self):
        """sparring3 should resolve to pro."""
        result = resolve_sparring_mode("sparring3")
        assert result == "pro", f"Expected 'pro', got '{result}'"
    
    def test_nor_alias_to_full(self):
        """nor should resolve to full."""
        result = resolve_sparring_mode("nor")
        assert result == "full", f"Expected 'full', got '{result}'"
    
    def test_flash_passthrough(self):
        """flash should pass through unchanged."""
        result = resolve_sparring_mode("flash")
        assert result == "flash", f"Expected 'flash', got '{result}'"
    
    def test_full_passthrough(self):
        """full should pass through unchanged."""
        result = resolve_sparring_mode("full")
        assert result == "full", f"Expected 'full', got '{result}'"
    
    def test_pro_passthrough(self):
        """pro should pass through unchanged."""
        result = resolve_sparring_mode("pro")
        assert result == "pro", f"Expected 'pro', got '{result}'"
    
    def test_invalid_mode_returns_flash(self):
        """Invalid mode should return flash as fallback."""
        result = resolve_sparring_mode("invalid_mode")
        assert result == "flash", f"Expected 'flash' as fallback, got '{result}'"


class TestSparringModeCaseInsensitive:
    """Test case-insensitive mode resolution."""
    
    def test_uppercase_sparring1(self):
        """SPARRING1 should resolve to flash."""
        result = resolve_sparring_mode("SPARRING1")
        assert result == "flash", f"Expected 'flash', got '{result}'"
    
    def test_mixed_case_sparring2(self):
        """SpArRiNg2 should resolve to full."""
        result = resolve_sparring_mode("SpArRiNg2")
        assert result == "full", f"Expected 'full', got '{result}'"
    
    def test_uppercase_flash(self):
        """FLASH should resolve to flash."""
        result = resolve_sparring_mode("FLASH")
        assert result == "flash", f"Expected 'flash', got '{result}'"
    
    def test_uppercase_pro(self):
        """PRO should resolve to pro."""
        result = resolve_sparring_mode("PRO")
        assert result == "pro", f"Expected 'pro', got '{result}'"
    
    def test_whitespace_handling(self):
        """'  sparring1  ' should resolve correctly."""
        result = resolve_sparring_mode("  sparring1  ")
        assert result == "flash", f"Expected 'flash', got '{result}'"


class TestSparringTimeoutConfiguration:
    """Test timeout configuration for each sparring level."""
    
    def test_flash_analyst_timeout_180s(self):
        """Flash analyst timeout should be 90s (180s total for 2 steps)."""
        assert TIMEOUTS["flash_analyst"] == 90.0, f"Expected 90.0, got {TIMEOUTS['flash_analyst']}"
    
    def test_flash_drafter_timeout_180s(self):
        """Flash drafter timeout should be 90s (180s total for 2 steps)."""
        assert TIMEOUTS["flash_drafter"] == 90.0, f"Expected 90.0, got {TIMEOUTS['flash_drafter']}"
    
    def test_discovery_timeout_100s(self):
        """Discovery timeout should be 100s for sparring3 step-by-step."""
        assert TIMEOUTS["discovery"] == 100.0, f"Expected 100.0, got {TIMEOUTS['discovery']}"
    
    def test_red_cell_timeout_100s(self):
        """Red cell timeout should be 100s for sparring3 step-by-step."""
        assert TIMEOUTS["red_cell"] == 100.0, f"Expected 100.0, got {TIMEOUTS['red_cell']}"
    
    def test_blue_cell_timeout_100s(self):
        """Blue cell timeout should be 100s for sparring3 step-by-step."""
        assert TIMEOUTS["blue_cell"] == 100.0, f"Expected 100.0, got {TIMEOUTS['blue_cell']}"
    
    def test_white_cell_timeout_100s(self):
        """White cell timeout should be 100s for sparring3 step-by-step."""
        assert TIMEOUTS["white_cell"] == 100.0, f"Expected 100.0, got {TIMEOUTS['white_cell']}"


class TestDefaultMode:
    """Test default mode configuration."""
    
    def test_default_mode_is_sparring2(self):
        """Default mode should be sparring2 (normal/full)."""
        # This test will verify the default mode in server.py
        # After implementation, default should be "sparring2" not "flash"
        # For now, we test that the function exists
        from qwen_mcp.tools import generate_sparring
        import inspect
        sig = inspect.signature(generate_sparring)
        mode_param = sig.parameters.get('mode')
        # After implementation, default should be 'sparring2'
        # Current default is 'flash', so this test will FAIL initially
        expected_default = "sparring2"
        actual_default = mode_param.default
        assert actual_default == expected_default, f"Expected default '{expected_default}', got '{actual_default}'"


class TestWordLimitInstruction:
    """Test word limit instruction exists in prompts."""
    
    def test_word_limit_instruction_exists(self):
        """WORD_LIMIT_INSTRUCTION should be defined in prompts/sparring.py."""
        from qwen_mcp.prompts.sparring import WORD_LIMIT_INSTRUCTION
        assert WORD_LIMIT_INSTRUCTION is not None
        assert "KOMPLETNA" in WORD_LIMIT_INSTRUCTION or "COMPLETE" in WORD_LIMIT_INSTRUCTION