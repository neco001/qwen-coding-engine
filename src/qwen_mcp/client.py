import os
import logging
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI, AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError, APIError
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecuritySanitizer:
    """Detection and redaction of sensitive patterns in strings."""
    
    # Common patterns for API keys, secrets, and PII
    PATTERNS = {
        "Generic API Key": r"(?i)(api[_-]?key|auth[_-]?token|secret|password|passwd|pwd)['\"]?\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{16,})['\"]?",
        "AWS Access Key": r"AKIA[0-9A-Z]{16}",
        "Generic Token": r"[tT]oken['\"]?\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{24,})['\"]?",
        "Email Address": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    }

    @classmethod
    def redact(cls, text: str) -> str:
        if not text:
            return text
        
        # Redaction can be disabled for local debugging
        if os.getenv("SECURITY_REDACTION_ENABLED", "false").lower() != "true":
            return text
            
        sanitized = text
        for name, pattern in cls.PATTERNS.items():
            matches = list(re.finditer(pattern, sanitized))
            if matches:
                logger.info(f"SecuritySanitizer: Found and redacting {len(matches)} potential {name}(s).")
                sanitized = re.sub(pattern, "[REDACTED]", sanitized)
        return sanitized

# Load environment variables
load_dotenv()

# Alibaba Cloud DashScope API base URL for OpenAI compatibility
DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

import json
from pathlib import Path
from datetime import datetime
from platformdirs import user_cache_dir

class ModelRegistry:
    """Dynamic registry for ROI-optimized model selection (JSON Cached)."""
    
    def __init__(self):
        # Force a clean path without nested "Cache" and "qwen-mcp" subfolders
        local_app_data = os.getenv("LOCALAPPDATA")
        if local_app_data:
            self.cache_dir = Path(local_app_data) / "qwen-mcp"
        else:
            self.cache_dir = Path.home() / ".cache" / "qwen-mcp"
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "models_cache.json"
        from .billing import billing_tracker # initialize db

        
        self.models = {
            "strategist": "qwen3.5-plus",
            "scout": "qwen-turbo",
            "coder": "qwen2.5-coder-32b-instruct"
        }
        self.last_updated = datetime.min
        self.load_cache()

    def load_cache(self):
        if self.cache_file.exists():
            try:
                data = json.loads(self.cache_file.read_text())
                self.models.update(data.get("models", {}))
                updated_at = data.get("updated_at")
                if updated_at:
                    self.last_updated = datetime.fromisoformat(updated_at)
                logger.info(f"ModelRegistry: Cache loaded (Last updated: {self.last_updated})")
            except Exception as e:
                logger.error(f"ModelRegistry: Failed to load cache: {e}")

    def is_stale(self, hours: int = 24) -> bool:
        """Checks if the cache is older than the specified hours."""
        from datetime import timedelta
        return (datetime.now() - self.last_updated) > timedelta(hours=hours)

    def save_cache(self):
        try:
            self.last_updated = datetime.now()
            data = {
                "updated_at": self.last_updated.isoformat(),
                "models": self.models
            }
            self.cache_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"ModelRegistry: Failed to save cache: {e}")

    def get_best_model(self, task_type: str) -> str:
        # Check for local Ollama override globally to avoid cloud model mismatch
        ollama_url = os.getenv("OLLAMA_BASE_URL")
        if ollama_url:
            # In local mode, we usually have one multi-purpose model
            return os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest")

        if task_type == "discovery":
            return self.models.get("scout", "qwen-turbo")
        if task_type == "coding":
            return self.models.get("coder", "qwen2.5-coder-32b-instruct")
        return self.models.get("strategist", "qwen3.5-plus")

    @property
    def STRATEGIST(self): return self.get_best_model("strategist")
    @property
    def SCOUT(self): return self.get_best_model("discovery")
    @property
    def CODER_SPECIALIST(self): return self.get_best_model("coding")

# Global Registry Instance
registry = ModelRegistry()

class DashScopeClient:
    """Client for interacting with Alibaba DashScope API and local Ollama instances."""
    
    def __init__(self):
        # Check for local Ollama override
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        
        if self.ollama_base_url:
            logger.info(f"OLLAMA_BASE_URL detected: {self.ollama_base_url}. Using local mode.")
            api_key = os.getenv("DASHSCOPE_API_KEY") or "ollama" 
            base_url = self.ollama_base_url
            # Default to qwen2.5-coder for local if not specified
            self.model_name = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest")
        elif os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIBABA_AI_KEY"):
            api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIBABA_AI_KEY")
            base_url = DASHSCOPE_BASE_URL
            self.model_name = os.getenv("QWEN_MODEL_NAME", "qwen-plus")
        else:
            logger.error("No Provider configured. Set DASHSCOPE_API_KEY for Cloud or OLLAMA_BASE_URL for Local.")
            raise ValueError("Configuration missing. Please check your .env or environment variables.")
        self.default_timeout = float(os.getenv("DASHSCOPE_TIMEOUT", "60.0"))
        
        # Initialize the OpenAI async client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=2
        )
        self.session_usage = {}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
        reraise=True
    )
    async def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.2,
        max_tokens: int = 4000,
        task_type: Optional[str] = None,
        timeout: Optional[float] = None,
        progress_callback = None
    ) -> str:
        """Generate a chat completion with robust retries, error handling, and optional progress streaming."""
        request_timeout = timeout or self.default_timeout
        
        # Sanitize all message content for security
        for msg in messages:
            if "content" in msg:
                msg["content"] = SecuritySanitizer.redact(msg["content"])

        for attempt in range(2):
            target_model = registry.get_best_model(task_type) if task_type else self.model_name
            if attempt == 0:
                logger.info(f"Requesting completion from {target_model} (timeout: {request_timeout}s)")
            else:
                logger.info(f"Retrying completion from updated model {target_model}...")

            try:
                if progress_callback:
                    # Streaming mode
                    response = await self.client.chat.completions.create(
                        model=target_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=request_timeout,
                        stream=True,
                        stream_options={"include_usage": True}
                    )
                    
                    full_response = ""
                    async for chunk in response:
                        # Capture token usage if available in the final chunk
                        if hasattr(chunk, "usage") and chunk.usage:
                            prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0)
                            completion_tokens = getattr(chunk.usage, "completion_tokens", 0)
                            if target_model not in self.session_usage:
                                self.session_usage[target_model] = {"prompt": 0, "completion": 0}
                            self.session_usage[target_model]["prompt"] += prompt_tokens
                            self.session_usage[target_model]["completion"] += completion_tokens
                            
                            # Log to persistent DuckDB Billing Tracker
                            from .billing import billing_tracker
                            project_name = os.getenv("QWEN_PROJECT_NAME", "adhoc")
                            # Wrap in try-except to avoid breaking generation on billing fails
                            try:
                                billing_tracker.log_usage(project_name, target_model, prompt_tokens, completion_tokens)
                            except Exception as e:
                                logger.error(f"Billing logging failed: {e}")

                        if hasattr(chunk, "choices") and chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta.content
                            if delta:
                                full_response += delta
                                await progress_callback(delta, len(full_response))
                    return full_response
                else:
                    # Standard mode
                    response = await self.client.chat.completions.create(
                        model=target_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=request_timeout,
                        stream=False
                    )
                    
                    # Capture token usage
                    if hasattr(response, "usage") and response.usage:
                        prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
                        completion_tokens = getattr(response.usage, "completion_tokens", 0)
                        if target_model not in self.session_usage:
                            self.session_usage[target_model] = {"prompt": 0, "completion": 0}
                        self.session_usage[target_model]["prompt"] += prompt_tokens
                        self.session_usage[target_model]["completion"] += completion_tokens

                        # Log to persistent DuckDB Billing Tracker
                        from .billing import billing_tracker
                        project_name = os.getenv("QWEN_PROJECT_NAME", "adhoc")
                        try:
                            billing_tracker.log_usage(project_name, target_model, prompt_tokens, completion_tokens)
                        except Exception as e:
                            logger.error(f"Billing logging failed: {e}")

                    if not response.choices or len(response.choices) == 0:
                        logger.warning("Empty choices from API.")
                        return "Error: Empty response."
                        
                    return response.choices[0].message.content or ""
            
            except Exception as e:
                is_model_error = "model_not_found" in str(e).lower() or "not found" in str(e).lower()
                if is_model_error and attempt == 0:
                    logger.warning(f"Model {target_model} rejected. Refreshing registry and retrying seamlessly...")
                    await self.refresh_registry()
                    continue
                
                if attempt == 1 or not is_model_error:
                    return await self._handle_error(e, request_timeout)

    async def generate_streaming_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4000,
        task_type: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        """Generate a streaming completion. Note: MCP tools typically don't support streaming directly yet, 
        but this serves as a foundation for future server-side features."""
        request_timeout = timeout or self.default_timeout
        
        for attempt in range(2):
            target_model = registry.get_best_model(task_type) if task_type else self.model_name
            try:
                stream = await self.client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=request_timeout,
                    stream=True
                )
                return stream
            except Exception as e:
                is_model_error = "model_not_found" in str(e).lower() or "not found" in str(e).lower()
                if is_model_error and attempt == 0:
                    await self.refresh_registry()
                    continue
                logger.error(f"Streaming error: {e}")
                raise

    async def probe_model(self, model_id: str) -> bool:
        """Pings the model with a minimal payload to ensure it is healthy and accessible."""
        try:
            logger.info(f"Probing candidate model: {model_id}...")
            # Set a very low timeout to prevent startup hangs during polling
            await self.client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "."}],
                max_tokens=1,
                timeout=2.0 
            )
            logger.info(f"Model probe successful: {model_id}")
            return True
        except Exception as e:
            logger.warning(f"Model probe failed for {model_id}: {e}")
            return False

    async def list_models(self) -> List[str]:
        """Fetch list of available models from DashScope."""
        try:
            response = await self.client.models.list()
            return [m.id for m in response.data]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    async def refresh_registry(self) -> str:
        """Fetch models, validate with 'Smoke Test' and update the registry cache."""
        models = await self.list_models()
        if not models:
            logger.warning("Dynamic discovery failed. Falling back to cached or hardcoded default models.")
            return "Failed to fetch models or API not responding."
        
        # Determine candidate list and probe until one works
        async def pick_best(preferred: List[str], all_models: List[str], default: str) -> str:
            candidates = []
            # Exact matches first
            for p in preferred:
                if p in all_models and p not in candidates: 
                    candidates.append(p)
            # Prefix matches second
            for p in preferred:
                for m in all_models:
                    if m.startswith(p) and m not in candidates: 
                        candidates.append(m)
            
            # Probe top candidates
            for candidate in candidates[:3]: # limit probing to top 3
                if await self.probe_model(candidate):
                    return candidate
            
            # If discovery completely failed or models are broken, return hardcoded constant
            logger.error(f"All candidates failed probing for {preferred}. Defaulting to hardcoded fallback: {default}")
            return default

        # The constants defined here serve as our HARDCODED_FALLBACKs
        # Strategist fallback -> "qwen-plus"
        # Coder fallback -> "qwen2.5-coder-32b-instruct"
        # Scout fallback -> "qwen-turbo"
        new_strategist = await pick_best(["qwen-max", "qwen-plus", "qwen3.5-plus"], models, "qwen-plus")
        new_coder = await pick_best(["qwen2.5-coder-32b-instruct", "qwen2.5-coder-7b-instruct"], models, "qwen2.5-coder-32b-instruct")
        new_scout = await pick_best(["qwen-turbo", "qwen-plus"], models, "qwen-turbo")

        updates = []
        if new_strategist and new_strategist != registry.models["strategist"]:
            registry.models["strategist"] = new_strategist
            updates.append(f"Strategist -> {new_strategist}")
        if new_coder and new_coder != registry.models["coder"]:
            registry.models["coder"] = new_coder
            updates.append(f"Coder -> {new_coder}")
        if new_scout and new_scout != registry.models["scout"]:
            registry.models["scout"] = new_scout
            updates.append(f"Scout -> {new_scout}")
        
        # Always save cache to update `last_updated` timestamp even if models are identical
        registry.save_cache()
        if updates:
            return f"Registry updated via probe validation: {', '.join(updates)}"
        return "Cache validated via probes. Current models are healthy and optimal."

    async def _handle_error(self, e: Exception, timeout: float) -> str:
        """Centralized error handling logic."""
        if "model_not_found" in str(e).lower() or "not found" in str(e).lower():
            return "Error: Model not found even after auto-refresh attempt. Please check your account permissions or configuration."
        
        if isinstance(e, AuthenticationError):
            logger.error("Authentication failed.")
            return "Error: Authentication failed."
        if isinstance(e, RateLimitError):
            logger.warning("Rate limit exceeded.")
            raise
        if isinstance(e, APITimeoutError):
            logger.error(f"Request timed out after {timeout}s.")
            raise
        if isinstance(e, APIConnectionError):
            logger.error("Connection error.")
            raise
        if isinstance(e, APIError):
            return f"Error: API Error ({e.status_code}): {e.message}"
        
        logger.exception("Unexpected error")
        return f"Error: {str(e)}"
