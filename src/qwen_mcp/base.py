import os
import logging
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv

from .sanitizer import SecuritySanitizer
from .registry import ModelRegistry, registry as global_registry

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Alibaba Cloud DashScope API base URL for OpenAI compatibility
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_API_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
# WanX Multi-modal API base URL (China region - handles Async better)
DASHSCOPE_WANX_BASE_URL = os.getenv("DASHSCOPE_WANX_BASE", "https://dashscope.aliyuncs.com/api/v1")

class BaseDashScopeClient:
    """Base client for DashScope/Ollama configuration and authentication."""

    def __init__(
        self, registry: Optional[ModelRegistry] = None, sanitizer_cls=SecuritySanitizer
    ):
        self.registry = registry or global_registry
        self.sanitizer_cls = sanitizer_cls

        # Check for local Ollama override
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL")

        if self.ollama_base_url:
            logger.info(f"OLLAMA_BASE_URL detected: {self.ollama_base_url}. Using local mode.")
            api_key = os.getenv("DASHSCOPE_API_KEY") or "ollama"
            base_url = self.ollama_base_url
            self.model_name = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest")
        elif os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIBABA_AI_KEY"):
            api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIBABA_AI_KEY")
            base_url = DASHSCOPE_BASE_URL
            self.model_name = os.getenv("QWEN_MODEL_NAME", "qwen-plus")
        else:
            msg = "No Provider configured. Set DASHSCOPE_API_KEY for Cloud or OLLAMA_BASE_URL for Local."
            logger.error(msg)
            raise ValueError(msg)

        self.default_timeout = float(os.getenv("DASHSCOPE_TIMEOUT", "60.0"))

        # Financial Guardrails
        self.max_input_tokens = int(os.getenv("MAX_INPUT_TOKENS", "32000"))
        self.max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "4000"))

        # Initialize the OpenAI async client
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=2)
        self.session_usage = {}

    def estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Pessimistic estimation: ~3 characters per token for code/text."""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars // 3

    def check_financial_circuit_breaker(self, messages: List[Dict[str, str]]) -> int:
        """Checks if input exceeds limits. Returns estimated tokens or raises ValueError."""
        estimated_input = self.estimate_tokens(messages)
        if estimated_input > self.max_input_tokens:
            msg = f"FINANCIAL CIRCUIT BREAKER: Input too large ({estimated_input} tokens). Limit is {self.max_input_tokens}."
            logger.error(msg)
            raise ValueError(msg)
        return estimated_input
