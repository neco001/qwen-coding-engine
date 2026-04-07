"""Tests for EAN13 checksum calculation."""
import os
import sys

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qwen_mcp.ean13 import calculate_ean13_checksum, validate_ean13_input


class TestEAN13Validation:
    """Test suite for EAN13 input validation."""

    def test_valid_12_digits_passes(self):
        """Given 12 digits, validation should pass."""
        assert validate_ean13_input("123456789012") is True

    def test_11_digits_fails(self):
        """Given 11 digits, validation should fail."""
        assert validate_ean13_input("12345678901") is False

    def test_13_digits_fails(self):
        """Given 13 digits, validation should fail."""
        assert validate_ean13_input("1234567890123") is False

    def test_non_digit_characters_fails(self):
        """Given non-digit characters, validation should fail."""
        assert validate_ean13_input("12345678901a") is False

    def test_empty_string_fails(self):
        """Given empty string, validation should fail."""
        assert validate_ean13_input("") is False

    def test_none_input_fails(self):
        """Given None, validation should fail."""
        assert validate_ean13_input(None) is False

    def test_special_characters_fails(self):
        """Given special characters, validation should fail."""
        assert validate_ean13_input("123-456-78901") is False

    def test_all_zeros_passes(self):
        """Given 12 zeros, validation should pass."""
        assert validate_ean13_input("000000000000") is True


class TestEAN13Checksum:
    """Test suite for EAN13 checksum calculation."""

    def test_valid_12_digits_returns_single_digit(self):
        """Given 12 valid digits, should return single check digit."""
        result = calculate_ean13_checksum("590123412345")
        assert isinstance(result, int)
        assert 0 <= result <= 9

    def test_known_ean13_checksum(self):
        """Given known EAN13 prefix, should return correct check digit."""
        # 5901234123457 is valid EAN13, check digit is 7
        result = calculate_ean13_checksum("590123412345")
        assert result == 7

    def test_another_known_checksum(self):
        """Given another known prefix, should return correct check digit."""
        # 4006381333931 is valid EAN13, check digit is 1
        result = calculate_ean13_checksum("400638133393")
        assert result == 1

    def test_all_zeros_returns_zero(self):
        """Given 12 zeros, should return 0 as check digit."""
        result = calculate_ean13_checksum("000000000000")
        assert result == 0

    def test_invalid_input_raises_value_error(self):
        """Given invalid input, should raise ValueError."""
        with pytest.raises(ValueError):
            calculate_ean13_checksum("123")

    def test_non_digit_input_raises_value_error(self):
        """Given non-digit input, should raise ValueError."""
        with pytest.raises(ValueError):
            calculate_ean13_checksum("12345678901a")

    def test_none_input_raises_value_error(self):
        """Given None, should raise ValueError."""
        with pytest.raises(ValueError):
            calculate_ean13_checksum(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
