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
DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
CODING_PLAN_BASE_URL = "https://coding-intl.dashscope.aliyuncs.com/v1"

# Global state for billing mode: 'payg', 'coding_plan', or 'hybrid'
ACTIVE_BILLING_MODE = os.getenv("BILLING_MODE", "coding_plan").lower()

CODING_PLAN_MODELS = {
    "qwen3.5-plus", "qwen3-coder-plus", "qwen3-coder-next",
    "qwen3-max-2026-01-23", "glm-4.7", "glm-5", "MiniMax-M2.5", "kimi-k2.5"
}

def set_billing_mode(mode: str) -> bool:
    global ACTIVE_BILLING_MODE
    mode = mode.lower()
    if mode in ["payg", "coding_plan", "hybrid"]:
        ACTIVE_BILLING_MODE = mode
        logger.info(f"Billing mode changed to: {mode}")
        return True
    return False

def get_billing_mode() -> str:
    return ACTIVE_BILLING_MODE

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
            self.api_key_payg = os.getenv("DASHSCOPE_API_KEY") or "ollama"
            base_url_payg = self.ollama_base_url
            self.model_name = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest")
        else:
            self.api_key_payg = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIBABA_AI_KEY")
            base_url_payg = DASHSCOPE_BASE_URL
            self.model_name = os.getenv("QWEN_MODEL_NAME", "qwen-plus")

        self.api_key_plan = os.getenv("CODING_PLAN_API_KEY") or os.getenv("BAILIAN_CODING_PLAN_API_KEY")
        if self.api_key_plan:
            self.api_key_plan = self.api_key_plan.strip('"').strip("'")

        if not self.api_key_payg and not self.api_key_plan:
            msg = "No Provider configured. Set DASHSCOPE_API_KEY for PAYG or CODING_PLAN_API_KEY for Coding Plan."
            logger.error(msg)
            raise ValueError(msg)

        self.default_timeout = float(os.getenv("DASHSCOPE_TIMEOUT", "60.0"))

        # Financial Guardrails
        self.max_input_tokens = int(os.getenv("MAX_INPUT_TOKENS", "32000"))
        self.max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))

        # Initialize the OpenAI async clients
        self.client_payg = AsyncOpenAI(api_key=self.api_key_payg, base_url=base_url_payg, max_retries=2) if self.api_key_payg else None
        self.client_plan = AsyncOpenAI(api_key=self.api_key_plan, base_url=CODING_PLAN_BASE_URL, max_retries=2) if self.api_key_plan else None
        
        self.active_warnings = []
        # Auto-fallback: If plan key is missing but mode requires it, downgrade to PAYG
        current_mode = get_billing_mode()
        if not self.client_plan and current_mode in ["coding_plan", "hybrid"]:
            msg = f"⚠️ WARNING: Coding Plan API key missing. Auto-falling back from '{current_mode}' to 'payg'."
            logger.warning(msg)
            set_billing_mode("payg")
            self.active_warnings.append(msg)

        # Legacy self.client attribute to avoid breaking existing usages, but internal methods will select dynamically.
        self.client = self.client_payg or self.client_plan
        self.session_usage = {}

    def get_client_for_model(self, model_id: str) -> AsyncOpenAI:
        """Returns the appropriate API client based on current BILLING_MODE and target model."""
        mode = get_billing_mode()
        
        if mode == "coding_plan":
            if not self.client_plan:
                msg = "⚠️ WARNING: Coding Plan API key not found. Falling back to PAYG client."
                logger.warning(msg)
                self.active_warnings.append(msg)
                return self.client_payg or self.client
            return self.client_plan
            
        elif mode == "payg":
            if not self.client_payg:
                msg = "⚠️ WARNING: PAYG API key not found. Falling back to Coding Plan client."
                logger.warning(msg)
                self.active_warnings.append(msg)
                return self.client_plan or self.client
            return self.client_payg
            
        elif mode == "hybrid":
            # Direct it to plan client if it's a plan-exclusive model or standard plan model, 
            # to save on PAYG costs where possible.
            if model_id in CODING_PLAN_MODELS and self.client_plan:
                return self.client_plan
            if self.client_payg:
                return self.client_payg
            return self.client_plan or self.client
            
        return self.client

    def estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Pessimistic estimation: ~3 characters per token for code/text."""
        total_chars = sum(len(m.get("content", "")) for m in messages if m.get("content"))
        return total_chars // 3

    def check_financial_circuit_breaker(self, messages: List[Dict[str, str]]) -> int:
        """Checks if input exceeds limits. Returns estimated tokens or raises ValueError."""
        # Note: Coding Plan does not charge per token, but we still apply context limits to avoid context window explosion
        estimated_input = self.estimate_tokens(messages)
        if estimated_input > self.max_input_tokens:
            msg = f"FINANCIAL CIRCUIT BREAKER: Input too large ({estimated_input} tokens). Limit is {self.max_input_tokens}."
            logger.error(msg)
            raise ValueError(msg)
        return estimated_input
