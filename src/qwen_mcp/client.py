import os
import logging
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI, AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError, APIError
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import re

# Configure logging
import asyncio
import json
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from platformdirs import user_cache_dir

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    """Dynamic registry for ROI-optimized model selection (JSON Cached)."""
    
    # Guidance for the Meta-Agent (Qwen-Turbo) when selecting models
    ROLE_PROMPTS = {
        "strategist": "Expert in high-level planning and JSON architecture. (Priority: qwen3.5-plus, qwen-plus)",
        "coder": "Production-grade code generation, long outputs. (Priority: qwen3-coder-plus, qwen-coder-plus)",
        "specialist": "Complex logic, algorithms, and refactoring specialist. (Priority: qwen3-coder-next, qwen2.5-coder-32b)",
        "analyst": "Deep reasoning, SRE audit, finding hidden bugs. (Priority: qwq-plus, qwq-32b)",
        "scout": "Fast, cheap, discovery, and summarization. (Priority: qwen-turbo, qwen-flash)"
    }

    def __init__(self):
        # Professional Cross-Platform Cache Pathing
        self.cache_dir = Path(user_cache_dir("qwen-coding", "Qwen"))
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Secure permissions: owner only
            if os.name != 'nt':
                os.chmod(self.cache_dir, 0o700)
        except Exception as e:
            logger.error(f"Failed to setup cache directory: {e}")
            
        self.cache_file = self.cache_dir / "models_cache.json"
        self._lock = asyncio.Lock()
        
        self.models = {
            "strategist": "qwen-plus",
            "coder": "qwen2.5-coder-32b-instruct",
            "specialist": "qwen2.5-coder-32b-instruct",
            "analyst": "qwen-plus",
            "scout": "qwen-turbo"
        }
        self.last_updated = datetime.min
        self.load_cache()

    def score_model(self, model_id: str, criteria: dict) -> int:
        """Heuristic scoring of a model ID against role criteria."""
        score = 0
        mid = model_id.lower()
        
        # Absolute garbage filter (multimedia/translations/audio)
        garbage = ["vl", "audio", "image", "tts", "asr", "vc", "omni", "mt", "realtime"]
        if any(g in mid for g in garbage) and not any(m in mid for m in criteria["must_have"]):
            return -500 # Heavy penalty for incompatible models

        # Key requirement weighting (High priority)
        for word in criteria["must_have"]:
            if word in mid: score += 100
        
        # Avoid criteria (Penalty)
        for word in criteria["avoid"]:
            if word in mid: score -= 50
        
        # Smart Version Analysis
        # Extract numbers but ignore 4-digit years (e.g., 2024, 2025, 2026)
        numbers = re.findall(r"(\d+(?:\.\d+)?)", mid)
        for num_str in numbers:
            num = float(num_str)
            if 1.0 <= num <= 10.0:  # Logical version numbers (Qwen 1.0 - 10.0)
                score += int(num * 10) # 3.5 becomes 35 points
            elif num > 2000: # It's likely a date suffix (e.g., -2025-01-01)
                score += 1 # Minor bonus for it being a recent snapshot, but doesn't override keywords

        # Stability preference
        if mid == criteria["fallback"]:
            score += 20
            
        # Specific penalty for snapshot/snapshot-date models if we prefer stable aliases
        if "-" in mid and any(char.isdigit() for char in mid.split("-")[-1]):
            score -= 5 # Prefer "qwen-plus" over "qwen-plus-2025..."

        return score

    def load_cache(self):
        if self.cache_file.exists():
            try:
                data = json.loads(self.cache_file.read_text())
                if data.get("schema_version", 1) > 1:
                    logger.warning("Cache schema version mismatch. Regenerating...")
                    return

                self.models.update(data.get("models", {}))
                updated_at = data.get("updated_at")
                if updated_at:
                    self.last_updated = datetime.fromisoformat(updated_at)
                logger.info("ModelRegistry: Cache loaded successfully.")
            except Exception as e:
                logger.error(f"ModelRegistry: Cache corrupted. Backing up and starting fresh: {e}")
                try:
                    self.cache_file.rename(self.cache_file.with_suffix(".json.bak"))
                except: pass

    async def save_cache(self):
        """Thread-safe, atomic write of the model cache."""
        async with self._lock:
            try:
                self.last_updated = datetime.now()
                data = {
                    "schema_version": 1,
                    "updated_at": self.last_updated.isoformat(),
                    "models": self.models
                }
                
                # Atomic Write Strategy (Write to tmp then move)
                temp_file = self.cache_file.with_suffix(".json.tmp")
                temp_file.write_text(json.dumps(data, indent=2))
                os.replace(temp_file, self.cache_file)
            except Exception as e:
                logger.error(f"ModelRegistry: Failed to save cache: {e}")

    def get_best_model(self, task_type: str) -> str:
        # Check for local Ollama override globally
        ollama_url = os.getenv("OLLAMA_BASE_URL")
        if ollama_url:
            return os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest")

        # Map task types to our ROI-optimized roles
        mapping = {
            "discovery": "scout",
            "scout": "scout",
            "coding": "coder",
            "coder": "coder",
            "refactoring": "specialist",
            "specialist": "specialist",
            "audit": "analyst",
            "analyst": "analyst",
            "planning": "strategist",
            "strategist": "strategist"
        }
        role = mapping.get(task_type, "strategist")
        return self.models.get(role, "qwen-plus")

    @property
    def STRATEGIST(self): return self.get_best_model("strategist")
    @property
    def SCOUT(self): return self.get_best_model("scout")
    @property
    def CODER_SPECIALIST(self): return self.get_best_model("coder")
    @property
    def LOGIC_SPECIALIST(self): return self.get_best_model("specialist")
    @property
    def ANALYST(self): return self.get_best_model("analyst")

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
        
        # Financial Guardrails
        self.max_input_tokens = int(os.getenv("MAX_INPUT_TOKENS", "32000"))
        self.max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "4000"))

        # Initialize the OpenAI async client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=2
        )
        self.session_usage = {}

    def estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Pessimistic estimation: ~3 characters per token for code/text."""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars // 3

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
        max_tokens: Optional[int] = None,
        task_type: Optional[str] = None,
        timeout: Optional[float] = None,
        progress_callback = None
    ) -> str:
        """Generate a chat completion with financial circuit breakers and retries."""
        request_timeout = timeout or self.default_timeout
        force_max_tokens = max_tokens or self.max_output_tokens

        # Financial Circuit Breaker: Input Check
        estimated_input = self.estimate_tokens(messages)
        if estimated_input > self.max_input_tokens:
            msg = f"FINANCIAL CIRCUIT BREAKER: Input too large ({estimated_input} tokens). Limit is {self.max_input_tokens}."
            logger.error(msg)
            return f"❌ {msg} Please truncate your files or context."
        
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
                        max_tokens=force_max_tokens,
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
                        max_tokens=force_max_tokens,
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
        max_tokens: Optional[int] = None,
        task_type: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        """Generate a streaming completion with financial guardrails."""
        request_timeout = timeout or self.default_timeout
        force_max_tokens = max_tokens or self.max_output_tokens

        # Financial Circuit Breaker: Input Check
        estimated_input = self.estimate_tokens(messages)
        if estimated_input > self.max_input_tokens:
            raise ValueError(f"FINANCIAL CIRCUIT BREAKER: Input too large ({estimated_input} tokens). Limit is {self.max_input_tokens}.")
        
        for attempt in range(2):
            target_model = registry.get_best_model(task_type) if task_type else self.model_name
            try:
                stream = await self.client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=force_max_tokens,
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

    async def probe_model(self, model_id: str, retries: int = 2) -> bool:
        """Pings the model with a minimal payload to ensure it is healthy. Retries with backoff."""
        for attempt in range(retries + 1):
            try:
                # Increasing timeout slightly for subsequent attempts
                timeout = 2.0 + (attempt * 1.5)
                logger.info(f"Probing candidate model: {model_id} (Attempt {attempt+1}/{retries+1})...")
                await self.client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    timeout=timeout
                )
                logger.info(f"Model probe successful: {model_id}")
                return True
            except Exception as e:
                if attempt == retries:
                    logger.warning(f"Model probe failed for {model_id} after {retries+1} attempts: {e}")
                    return False
                wait_time = 0.5 * (2 ** attempt)
                await asyncio.sleep(wait_time)
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
        """Alias for heal_registry to provide seamless transition to LLM-driven selection."""
        return await self.heal_registry()

    async def heal_registry(self) -> str:
        """Advanced Self-Healing: Uses meta-analysis via qwen-turbo to map roles by ROI."""
        logger.info("Initiating Self-Healing Meta-Analysis via Qwen-Turbo...")
        all_models = await self.list_models()
        if not all_models:
            return "Self-healing failed: Could not fetch model list from API."

        # Filter out obvious multimedia noise to keep prompt clean
        noise = ["audio", "tts", "asr", "vc", "image", "omni", "mt", "realtime"]
        clean_list = [m for m in all_models if not any(x in m.lower() for x in noise)]
        
        roles_desc = "\n".join([f"- {r}: {d}" for r, d in registry.ROLE_PROMPTS.items()])
        
        prompt = f"""You are the Qwen Project Administrator. Your goal is ROI optimization.
Assign the BEST model ID from the list below to each required role.
PRIORITIZE:
1. Stability: Prefer pure aliases (e.g. 'qwen-plus') over specific snapshot dates (e.g. 'qwen-plus-2025...') unless the date model is a clear generation jump (e.g. Qwen 3.5 vs 2.5).
2. Capability: Matching the role description to the model specialty.

Roles to assign:
{roles_desc}

AVAILABLE MODELS:
{clean_list}

Return ONLY a JSON object: {{"strategist": "id", "coder": "id", "specialist": "id", "analyst": "id", "scout": "id"}}"""

        try:
            # We use the known stable 'qwen-turbo' for the meta-analysis itself
            response = await self.client.chat.completions.create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
                timeout=20.0
            )
            
            from .tools import extract_json_from_text
            new_models = extract_json_from_text(response.choices[0].message.content)
            
            if new_models and isinstance(new_models, dict):
                # Validation: ensure selected models actually exist and are healthy (limited probe)
                updates = []
                for role, m_id in new_models.items():
                    if m_id in all_models:
                        # Probe only if it changed to avoid redundant testing
                        if m_id != registry.models.get(role):
                            if await self.probe_model(m_id):
                                registry.models[role] = m_id
                                updates.append(f"{role.capitalize()} -> {m_id}")
                            else:
                                logger.warning(f"Self-healing: Model {m_id} for {role} failed probe. Skipping.")
                    else:
                        logger.warning(f"Self-healing: Model {m_id} not in API list. Skipping.")

                await registry.save_cache()
                if updates:
                    return f"Registry healed via Qwen-Turbo Meta-Analysis! Updates: {', '.join(updates)}"
                return "Registry validated. All LLM-selected models are healthy and optimal."
            
            return "Self-healing failed: LLM response was not valid JSON."
            
        except Exception as e:
            logger.error(f"Self-healing error: {e}")
            return f"Self-healing failed: {str(e)}"

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
