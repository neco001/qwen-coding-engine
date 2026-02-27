import os
import asyncio
import logging
import codecs
from typing import List, Dict, Any, Optional

import textwrap
from openai import (
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .sanitizer import SecuritySanitizer
from .registry import ModelRegistry, registry as global_registry
from .specter.telemetry import get_broadcaster

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Alibaba Cloud DashScope API base URL for OpenAI compatibility
DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class DashScopeClient:
    """Client for interacting with Alibaba DashScope API and local Ollama instances."""

    def __init__(
        self, registry: Optional[ModelRegistry] = None, sanitizer_cls=SecuritySanitizer
    ):
        self.registry = registry or global_registry
        self.sanitizer_cls = sanitizer_cls

        # Check for local Ollama override
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL")

        if self.ollama_base_url:
            logger.info(
                f"OLLAMA_BASE_URL detected: {self.ollama_base_url}. Using local mode."
            )
            api_key = os.getenv("DASHSCOPE_API_KEY") or "ollama"
            base_url = self.ollama_base_url
            # Default to qwen2.5-coder for local if not specified
            self.model_name = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest")
        elif os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIBABA_AI_KEY"):
            api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIBABA_AI_KEY")
            base_url = DASHSCOPE_BASE_URL
            self.model_name = os.getenv("QWEN_MODEL_NAME", "qwen-plus")
        else:
            logger.error(
                "No Provider configured. Set DASHSCOPE_API_KEY for Cloud or OLLAMA_BASE_URL for Local."
            )
            raise ValueError(
                "Configuration missing. Please check your .env or environment variables."
            )

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIConnectionError)
        ),
        reraise=True,
    )
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        task_type: Optional[str] = None,
        timeout: Optional[float] = None,
        progress_callback=None,
        complexity: str = "auto",
        tags: Optional[List[str]] = None,
        include_reasoning: bool = False,
        model_override: Optional[str] = None,
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
                msg["content"] = self.sanitizer_cls.redact(msg["content"])

        for attempt in range(2):
            target_model = model_override or (
                self.registry.route_request(
                    task_type, complexity_hint=complexity, context_tags=tags or []
                )
                if task_type
                else self.model_name
            )
            if attempt == 0:
                logger.warning(
                    f"DEBUG: Requesting completion from model ID: '{target_model}' (override: {model_override})"
                )
            else:
                logger.info(f"Retrying completion from updated model {target_model}...")

            # Auto-detect reasoning models (qwq-* requires streaming + enable_thinking)
            is_reasoning_model = "qwq" in target_model.lower() or task_type == "analyst"
            use_streaming = bool(progress_callback) or is_reasoning_model
            extra_body = {"enable_thinking": True} if is_reasoning_model else {}

            try:
                if use_streaming:
                    # Streaming mode (required for reasoning models)
                    response = await self.client.chat.completions.create(
                        model=target_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=force_max_tokens,
                        timeout=request_timeout,
                        stream=True,
                        stream_options={"include_usage": True},
                        extra_body=extra_body if extra_body else None,
                    )

                    full_response = ""
                    reasoning_log = ""
                    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                    
                    async for chunk in response:
                        # ... (usage tracking remains same)
                        if hasattr(chunk, "usage") and chunk.usage:
                            prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0)
                            completion_tokens = getattr(
                                chunk.usage, "completion_tokens", 0
                            )
                            if target_model not in self.session_usage:
                                self.session_usage[target_model] = {
                                    "prompt": 0,
                                    "completion": 0,
                                }
                            self.session_usage[target_model]["prompt"] += prompt_tokens
                            self.session_usage[target_model]["completion"] += (
                                completion_tokens
                            )

                            # Log to persistent DuckDB Billing Tracker
                            from .billing import billing_tracker

                            project_name = os.getenv("QWEN_PROJECT_NAME", "adhoc")
                            try:
                                billing_tracker.log_usage(
                                    project_name,
                                    target_model,
                                    prompt_tokens,
                                    completion_tokens,
                                )
                                # Broadcast to HUD
                                await get_broadcaster().report_usage(
                                    target_model, prompt_tokens, completion_tokens
                                )
                            except Exception as e:
                                logger.error(f"Telemetry/Billing logging failed: {e}")

                        if hasattr(chunk, "choices") and chunk.choices:
                            delta = chunk.choices[0].delta

                            # Support for reasoning_content (QwQ models)
                            # Note: DashScope's 'reasoning_content' often comes in chunks.
                            # We treat both content and reasoning_content as potential sources.
                            
                            reasoning = getattr(delta, "reasoning_content", None)
                            if reasoning:
                                reasoning_log += reasoning
                                # Broadast to HUD directly (safe, bypassed Pydantic/MCP validation)
                                await get_broadcaster().update_stream(thinking=reasoning)

                            content = getattr(delta, "content", None)
                            if content:
                                full_response += content
                                # Broadcast content to HUD
                                await get_broadcaster().update_stream(content=content)
                                
                                # MCP Context progress is too sensitive for high-frequency chunks
                                # we keep it disabled here to avoid Validation Errors.

                    if reasoning_log:
                        logger.info(
                            f"Reasoning completed ({len(reasoning_log)} chars)."
                        )

                    if include_reasoning and reasoning_log:
                        return f"<thought>\n{reasoning_log.strip()}\n</thought>\n\n{full_response.strip()}".strip()
                    
                    return (full_response or reasoning_log).strip()
                else:
                    # Standard mode
                    response = await self.client.chat.completions.create(
                        model=target_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=force_max_tokens,
                        timeout=request_timeout,
                        stream=False,
                    )

                    # Capture token usage
                    if hasattr(response, "usage") and response.usage:
                        prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
                        completion_tokens = getattr(
                            response.usage, "completion_tokens", 0
                        )
                        if target_model not in self.session_usage:
                            self.session_usage[target_model] = {
                                "prompt": 0,
                                "completion": 0,
                            }
                        self.session_usage[target_model]["prompt"] += prompt_tokens
                        self.session_usage[target_model]["completion"] += (
                            completion_tokens
                        )

                        # Log to persistent DuckDB Billing Tracker
                        from .billing import billing_tracker

                        project_name = os.getenv("QWEN_PROJECT_NAME", "adhoc")
                        try:
                            billing_tracker.log_usage(
                                project_name,
                                target_model,
                                prompt_tokens,
                                completion_tokens,
                            )
                            # Broadcast to HUD
                            await get_broadcaster().report_usage(
                                target_model, prompt_tokens, completion_tokens
                            )
                        except Exception as e:
                            logger.error(f"Telemetry/Billing logging failed: {e}")

                    if not response.choices or len(response.choices) == 0:
                        logger.warning("Empty choices from API.")
                        return "Error: Empty response."

                    msg = response.choices[0].message
                    # Priority 1: standard content, Priority 2: reasoning_content (fallback)
                    content = getattr(msg, "content", "") or ""
                    reasoning = getattr(msg, "reasoning_content", "") or ""

                    if include_reasoning and reasoning:
                        return f"<thought>\n{reasoning.strip()}\n</thought>\n\n{content.strip()}".strip()

                    final_text = (content or reasoning).strip()
                    if not final_text:
                        logger.error("API returned empty content AND empty reasoning.")
                        return "Error: Empty response content."

                    return final_text

            except Exception as e:
                is_model_error = (
                    "model_not_found" in str(e).lower() or "not found" in str(e).lower()
                )
                if is_model_error and attempt == 0:
                    logger.warning(
                        f"Model {target_model} rejected. Refreshing registry and retrying seamlessly..."
                    )
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
        timeout: Optional[float] = None,
        complexity: str = "auto",
        tags: Optional[List[str]] = None,
    ):
        """Generate a streaming completion with financial guardrails."""
        request_timeout = timeout or self.default_timeout
        force_max_tokens = max_tokens or self.max_output_tokens

        if task_type:
            logger.info(
                f"API Request | Task: {task_type} | Models in Registry: {len(self.registry.metadata)}"
            )

        # Log input preview
        if messages:
            last_msg = messages[-1].get("content", "")[:100].replace("\n", " ")
            logger.info(
                f"API Input | Messages: {len(messages)} | Last User Msg: {last_msg}..."
            )

        # Financial Circuit Breaker: Input Check
        estimated_input = self.estimate_tokens(messages)
        if estimated_input > self.max_input_tokens:
            raise ValueError(
                f"FINANCIAL CIRCUIT BREAKER: Input too large ({estimated_input} tokens). Limit is {self.max_input_tokens}."
            )

        for attempt in range(2):
            target_model = (
                self.registry.route_request(
                    task_type, complexity_hint=complexity, context_tags=tags or []
                )
                if task_type
                else self.model_name
            )
            try:
                stream = await self.client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=force_max_tokens,
                    timeout=request_timeout,
                    stream=True,
                )
                return stream
            except Exception as e:
                is_model_error = (
                    "model_not_found" in str(e).lower() or "not found" in str(e).lower()
                )
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
                logger.info(
                    f"Probing candidate model: {model_id} (Attempt {attempt + 1}/{retries + 1})..."
                )
                await self.client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    timeout=timeout,
                )
                logger.info(f"Model probe successful: {model_id}")
                return True
            except Exception as e:
                if attempt == retries:
                    logger.warning(
                        f"Model probe failed for {model_id} after {retries + 1} attempts: {e}"
                    )
                    return False
                wait_time = 0.5 * (2**attempt)
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

        roles_desc = "\n".join(
            [f"- {r}: {d}" for r, d in self.registry.ROLE_PROMPTS.items()]
        )

        prompt = textwrap.dedent(f"""\
            You are the Qwen Project Administrator. Your goal is ROI optimization.
            Assign the BEST model ID from the list below to each required role.
            PRIORITIZE:
            1. SOTA (Generation Jump): ALWAYS prefer 'qwen3.5' models over 'qwen2.5' or generic 'qwen-plus/turbo' if available. 3.5 is the current gold standard.
            2. Stability: Prefer pure aliases (e.g. 'qwen3.5-plus') over specific snapshot dates (e.g. 'qwen3.5-plus-2025...') unless no pure alias exists.
            3. Capability: Match the role description to the model specialty (e.g. 'coder' role -> 'qwen3-coder-plus').
            
            Roles to assign:
            {roles_desc}
            
            AVAILABLE MODELS:
            {clean_list}
            
            Return ONLY a JSON object: {{"strategist": "id", "coder": "id", "specialist": "id", "analyst": "id", "scout": "id"}}""")

        try:
            # We use the known stable 'qwen-turbo' for the meta-analysis itself
            response = await self.client.chat.completions.create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
                timeout=20.0,
            )

            from .tools import extract_json_from_text

            new_models = extract_json_from_text(response.choices[0].message.content)

            if new_models and isinstance(new_models, dict):
                # Validation: ensure selected models actually exist and are healthy (limited probe)
                updates = []
                for role, m_id in new_models.items():
                    if m_id in all_models:
                        # Probe only if it changed to avoid redundant testing
                        if m_id != self.registry.models.get(role):
                            if await self.probe_model(m_id):
                                self.registry.models[role] = m_id
                                updates.append(f"{role.capitalize()} -> {m_id}")
                            else:
                                logger.warning(
                                    f"Self-healing: Model {m_id} for {role} failed probe. Skipping."
                                )
                    else:
                        logger.warning(
                            f"Self-healing: Model {m_id} not in API list. Skipping."
                        )

                await self.registry.save_cache()
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
            return f"Error: Model not found. Exception: {str(e)}"

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
