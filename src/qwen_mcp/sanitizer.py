import os
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SecuritySanitizer:
    """Detection and redaction of sensitive patterns in strings."""

    PATTERNS = {
        "Generic API Key": r"(?i)(api[_-]?key|auth[_-]?token|secret|password|passwd|pwd)['\"]?\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{16,})['\"]?",
        "AWS Access Key": r"AKIA[0-9A-Z]{16}",
        "Generic Token": r"[tT]oken['\"]?\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{24,})['\"]?",
        "Email Address": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    }

    @classmethod
    def redact(cls, text: str) -> str:
        if not text:
            return text

        if os.getenv("SECURITY_REDACTION_ENABLED", "false").lower() != "true":
            return text

        sanitized = text
        for name, pattern in cls.PATTERNS.items():
            matches = list(re.finditer(pattern, sanitized))
            if matches:
                logger.info(
                    f"SecuritySanitizer: Found and redacting {len(matches)} potential {name}(s)."
                )
                sanitized = re.sub(pattern, "[REDACTED]", sanitized)
        return sanitized


class ContentValidator:
    """Sanitization and validation of LLM inputs and outputs."""

    MAX_INPUT_LENGTH = 10000
    MIN_RESPONSE_LENGTH = 10

    @classmethod
    def sanitize_input(cls, text: str) -> str:
        """Truncate, strip control chars, and normalize whitespace."""
        if not text:
            return ""

        # Truncate to safe max
        text = text[: cls.MAX_INPUT_LENGTH]

        # Remove control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Normalize whitespace
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def validate_response(cls, response: Optional[str]) -> str:
        """Ensure response is valid and non-generic."""
        if not response or not isinstance(response, str):
            raise ValueError(
                f"Response is empty or invalid type. Got: {type(response)} -> {str(response)[:100]}"
            )

        response = response.strip()

        if len(response) < cls.MIN_RESPONSE_LENGTH:
            raise ValueError(
                f"Response too short (min {cls.MIN_RESPONSE_LENGTH} chars)."
            )

        # Block generic AI boilerplate
        generic_patterns = [
            r"^[Ii] can(?:'t| not)?",
            r"^[Aa]s an AI",
            r"^[Ii] am unable",
            r"^[Ss]orry",
        ]
        for pattern in generic_patterns:
            if re.match(pattern, response):
                raise ValueError("Response rejected as generic AI boilerplate.")

        return response
