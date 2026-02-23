import os
import re
import logging

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
