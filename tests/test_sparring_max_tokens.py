"""
Test weryfikujący konfigurację MAX_TOKENS_CONFIG i MAX_THINKING_TOKENS_CONFIG dla sparring engine v2.

Testy sprawdzają:
1. Czy MAX_TOKENS_CONFIG istnieje i ma poprawną strukturę
2. Czy get_max_tokens_for_step() zwraca poprawne wartości
3. Czy konfiguracja jest spójna z timeoutami
4. Czy MAX_THINKING_TOKENS_CONFIG istnieje i ma poprawne wartości dla sparring1/2/3
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qwen_mcp.engines.sparring_v2.config import (
    MAX_TOKENS_CONFIG,
    MAX_THINKING_TOKENS_CONFIG,
    get_max_tokens_for_step,
    get_thinking_tokens_for_mode,
    TIMEOUTS
)


class TestMaxTokensConfig:
    """Testy dla konfiguracji max_tokens."""
    
    def test_max_tokens_config_exists(self):
        """RED: MAX_TOKENS_CONFIG musi istnieć."""
        assert MAX_TOKENS_CONFIG is not None
        assert isinstance(MAX_TOKENS_CONFIG, dict)
    
    def test_max_tokens_config_has_all_modes(self):
        """RED: MAX_TOKENS_CONFIG musi mieć wszystkie tryby."""
        required_modes = ["flash", "full", "pro"]
        for mode in required_modes:
            assert mode in MAX_TOKENS_CONFIG, f"Brak trybu '{mode}' w MAX_TOKENS_CONFIG"
    
    def test_flash_mode_tokens(self):
        """RED: sparring1 (flash) musi mieć 1024 tokenów."""
        assert MAX_TOKENS_CONFIG["flash"]["analyst"] == 1024
        assert MAX_TOKENS_CONFIG["flash"]["drafter"] == 1024
    
    def test_full_mode_tokens(self):
        """RED: sparring2 (full) musi mieć poprawne tokeny."""
        assert MAX_TOKENS_CONFIG["full"]["discovery"] == 512
        assert MAX_TOKENS_CONFIG["full"]["red"] == 1024
        assert MAX_TOKENS_CONFIG["full"]["blue"] == 1024
        assert MAX_TOKENS_CONFIG["full"]["white"] == 1024
    
    def test_pro_mode_tokens(self):
        """RED: sparring3 (pro) musi mieć 4096 tokenów."""
        assert MAX_TOKENS_CONFIG["pro"]["discovery"] == 512
        assert MAX_TOKENS_CONFIG["pro"]["red"] == 4096
        assert MAX_TOKENS_CONFIG["pro"]["blue"] == 4096
        assert MAX_TOKENS_CONFIG["pro"]["white"] == 4096
    
    def test_get_max_tokens_for_step_flash(self):
        """RED: get_max_tokens_for_step musi działać dla flash."""
        assert get_max_tokens_for_step("flash", "analyst") == 1024
        assert get_max_tokens_for_step("flash", "drafter") == 1024
    
    def test_get_max_tokens_for_step_full(self):
        """RED: get_max_tokens_for_step musi działać dla full."""
        assert get_max_tokens_for_step("full", "discovery") == 512
        assert get_max_tokens_for_step("full", "red") == 1024
        assert get_max_tokens_for_step("full", "blue") == 1024
        assert get_max_tokens_for_step("full", "white") == 1024
    
    def test_get_max_tokens_for_step_pro(self):
        """RED: get_max_tokens_for_step musi działać dla pro."""
        assert get_max_tokens_for_step("pro", "discovery") == 512
        assert get_max_tokens_for_step("pro", "red") == 4096
        assert get_max_tokens_for_step("pro", "blue") == 4096
        assert get_max_tokens_for_step("pro", "white") == 4096
    
    def test_get_max_tokens_for_step_default(self):
        """RED: get_max_tokens_for_step musi zwracać default dla nieznanych."""
        assert get_max_tokens_for_step("unknown", "step") == 2048
        assert get_max_tokens_for_step("flash", "unknown") == 2048
    
    def test_total_tokens_calculation(self):
        """RED: Suma tokenów musi być poprawna."""
        # sparring1: 1024 + 1024 = 2048
        flash_total = (
            MAX_TOKENS_CONFIG["flash"]["analyst"] +
            MAX_TOKENS_CONFIG["flash"]["drafter"]
        )
        assert flash_total == 2048
        
        # sparring2: 512 + 1024*3 = 3584
        full_total = (
            MAX_TOKENS_CONFIG["full"]["discovery"] +
            MAX_TOKENS_CONFIG["full"]["red"] +
            MAX_TOKENS_CONFIG["full"]["blue"] +
            MAX_TOKENS_CONFIG["full"]["white"]
        )
        assert full_total == 3584
        
        # sparring3: 512 + 4096*3 = 12800
        pro_total = (
            MAX_TOKENS_CONFIG["pro"]["discovery"] +
            MAX_TOKENS_CONFIG["pro"]["red"] +
            MAX_TOKENS_CONFIG["pro"]["blue"] +
            MAX_TOKENS_CONFIG["pro"]["white"]
        )
        assert pro_total == 12800


class TestMaxThinkingTokensConfig:
    """Testy dla konfiguracji max_thinking_tokens."""
    
    def test_max_thinking_tokens_config_exists(self):
        """RED: MAX_THINKING_TOKENS_CONFIG musi istnieć."""
        assert MAX_THINKING_TOKENS_CONFIG is not None
        assert isinstance(MAX_THINKING_TOKENS_CONFIG, dict)
    
    def test_max_thinking_tokens_config_has_all_modes(self):
        """RED: MAX_THINKING_TOKENS_CONFIG musi mieć wszystkie tryby."""
        required_modes = ["sparring1", "sparring2", "sparring3"]
        for mode in required_modes:
            assert mode in MAX_THINKING_TOKENS_CONFIG, f"Brak trybu '{mode}' w MAX_THINKING_TOKENS_CONFIG"
    
    def test_sparring1_thinking_tokens(self):
        """RED: sparring1 (flash) musi mieć 512 thinking tokens."""
        assert MAX_THINKING_TOKENS_CONFIG["sparring1"] == 512
    
    def test_sparring2_thinking_tokens(self):
        """RED: sparring2 (full) musi mieć 256 thinking tokens (fixes timeout)."""
        assert MAX_THINKING_TOKENS_CONFIG["sparring2"] == 256
    
    def test_sparring3_thinking_tokens(self):
        """RED: sparring3 (pro) musi mieć 1024 thinking tokens."""
        assert MAX_THINKING_TOKENS_CONFIG["sparring3"] == 1024
    
    def test_get_thinking_tokens_for_mode_sparring1(self):
        """RED: get_thinking_tokens_for_mode musi zwracać 512 dla sparring1."""
        assert get_thinking_tokens_for_mode("sparring1") == 512
    
    def test_get_thinking_tokens_for_mode_sparring2(self):
        """RED: get_thinking_tokens_for_mode musi zwracać 256 dla sparring2."""
        assert get_thinking_tokens_for_mode("sparring2") == 256
    
    def test_get_thinking_tokens_for_mode_sparring3(self):
        """RED: get_thinking_tokens_for_mode musi zwracać 1024 dla sparring3."""
        assert get_thinking_tokens_for_mode("sparring3") == 1024
    
    def test_get_thinking_tokens_for_mode_default(self):
        """RED: get_thinking_tokens_for_mode musi zwracać default dla nieznanych."""
        assert get_thinking_tokens_for_mode("unknown") == 256
    
    def test_thinking_tokens_timeout_alignment(self):
        """RED: Thinking tokens muszą być zgodne z timeoutami."""
        # sparring1: 180s total, 512 thinking tokens per step
        # sparring2: 180s total, 256 thinking tokens per step (reduced to fix timeout)
        # sparring3: 100s per step, 1024 thinking tokens per step
        assert MAX_THINKING_TOKENS_CONFIG["sparring1"] == 512
        assert MAX_THINKING_TOKENS_CONFIG["sparring2"] == 256
        assert MAX_THINKING_TOKENS_CONFIG["sparring3"] == 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
