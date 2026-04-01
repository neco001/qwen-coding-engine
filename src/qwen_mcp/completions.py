import os
import logging
import asyncio
import codecs
import textwrap
from typing import List, Dict, Any, Optional

from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .base import BaseDashScopeClient, get_billing_mode
from .specter.telemetry import get_broadcaster
from .billing import billing_tracker

logger = logging.getLogger(__name__)

class CompletionHandler(BaseDashScopeClient):
    """Handles chat and streaming completions with retry logic and registry healing."""

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
        enable_thinking: Optional[bool] = None,
        max_thinking_tokens: Optional[int] = None,
        project_id: str = "default",
    ) -> str:
        """Generate common chat completion with financial circuit breakers and retries.
        
        Args:
            enable_thinking: Override thinking mode. None = auto-detect for reasoning models.
            max_thinking_tokens: Limit thinking tokens for deep-thinking models (default: 2048).
            project_id: Project/session ID for telemetry isolation (format: {instance}_{source}_{hash}).
        """
        request_timeout = timeout or self.default_timeout
        
        # Dynamic max_tokens based on complexity (controls response length & time)
        if max_tokens:
            force_max_tokens = max_tokens
        elif complexity == "critical":
            force_max_tokens = 6144  # Deep architecture/multi-file (Strategic mid-point)
        elif complexity == "high":
            force_max_tokens = 4096  # Standard features
        elif complexity == "medium":
            force_max_tokens = 2048  # Quick fixes
        else:  # low/auto
            force_max_tokens = 1024   # Fast, focused responses

        # Check circuit breaker
        try:
            self.check_financial_circuit_breaker(messages)
        except ValueError as e:
            return f"❌ {str(e)} Please truncate your files or context."

        # Sanitize all message content for security
        for msg in messages:
            if "content" in msg:
                msg["content"] = self.sanitizer_cls.redact(msg["content"])

        for attempt in range(2):
            billing_mode_val = get_billing_mode()
            estimated_input = self.estimate_tokens(messages) or 0
            target_model = model_override or (
                self.registry.route_request(
                    task_type, 
                    complexity_hint=complexity, 
                    context_tags=tags or [], 
                    billing_mode=billing_mode_val,
                    estimated_tokens=estimated_input
                )
                if task_type
                else self.model_name
            )
            
            # Use provided project_id (passed from caller with client_source already resolved)
            logger.info(f"Completion Request | Model: {target_model} | Attempt: {attempt+1} | Project: {project_id}")

            # Telemetry: Update total tokens in HUD with estimated plan tokens
            await get_broadcaster().broadcast_state({
                "active_model": target_model,
                "request_tokens": {"prompt": estimated_input, "completion": 0}
            }, project_id=project_id)

            # Auto-detect reasoning/deep-thinking models
            # Models with deep thinking capabilities that may need budget limits
            deep_thinking_models = ["glm-5", "glm-4.7", "qwen3-max", "qwen3.5-plus", "qwq"]
            is_deep_thinking = any(m in target_model.lower() for m in deep_thinking_models)
            is_reasoning_model = "qwq" in target_model.lower() or task_type == "analyst"
            use_streaming = bool(progress_callback) or is_reasoning_model or is_deep_thinking
            
            # Build extra_body for thinking mode control
            extra_body = {}
            if enable_thinking is True or (enable_thinking is None and is_deep_thinking):
                extra_body["enable_thinking"] = True
                # Dynamic thinking budget based on complexity
                # - low/medium: 1024 tokens (fast, focused)
                # - high: 2048 tokens (balanced)
                # - critical: 4096 tokens (deep analysis)
                if max_thinking_tokens:
                    thinking_budget = max_thinking_tokens
                elif complexity == "critical":
                    thinking_budget = 4096
                elif complexity == "high":
                    thinking_budget = 2048
                else:  # low, medium, auto
                    thinking_budget = 1024
                extra_body["max_thinking_tokens"] = thinking_budget
                logger.info(f"Thinking mode enabled for {target_model} with budget {thinking_budget} tokens (complexity: {complexity})")

            try:
                if use_streaming:
                    return await self._stream_completion(
                        target_model, messages, temperature, force_max_tokens,
                        request_timeout, extra_body, include_reasoning, project_id=project_id,
                        progress_callback=progress_callback
                    )
                else:
                    return await self._standard_completion(
                        target_model, messages, temperature, force_max_tokens,
                        request_timeout, include_reasoning, project_id=project_id
                    )
            except Exception as e:
                is_model_error = "model_not_found" in str(e).lower() or "not found" in str(e).lower()
                if is_model_error and attempt == 0:
                    logger.warning(f"Model {target_model} rejected. Refreshing registry...")
                    await self.heal_registry()
                    continue
                return await self._handle_error(e, request_timeout)

    async def _stream_completion(self, model, messages, temp, max_t, timeout, extra, include_reasoning, project_id="default", progress_callback=None):
        client_to_use = self.get_client_for_model(model)
        
        if progress_callback:
            try:
                await progress_callback(progress=5.0, message=f"Connecting to {model}...")
            except Exception:
                pass

        response = await client_to_use.chat.completions.create(
            model=model, messages=messages, temperature=temp, max_tokens=max_t,
            timeout=timeout, stream=True, stream_options={"include_usage": True},
            extra_body=extra if extra else None,
        )

        if progress_callback:
            try:
                await progress_callback(progress=10.0, message=f"Connection established, waiting for model...")
            except Exception:
                pass

        full_response = ""
        reasoning_log = ""
        usage_reported = False
        chunk_count = 0

        # Create event to control heartbeat
        first_chunk_received = asyncio.Event()

        # Heartbeat task to prevent MCP client timeout during model 'thinking' phase
        async def heartbeat_loop():
            # Wait a few seconds before starting heartbeats
            await asyncio.sleep(15)
            while not first_chunk_received.is_set():
                if progress_callback:
                    try:
                        await progress_callback(progress=15.0, message="Model is deep-thinking (reasoning phase)...")
                    except Exception:
                        pass
                await asyncio.sleep(15)

        heartbeat_task = asyncio.create_task(heartbeat_loop())

        try:
            async for chunk in response:
                if not first_chunk_received.is_set():
                    first_chunk_received.set()
                
                chunk_count += 1
                
                # CRITICAL: Call progress_callback every 10 chunks to keep MCP connection alive
                # This prevents MCP timeout -32001 during long streaming operations
                if progress_callback and chunk_count % 10 == 0:
                    try:
                        await progress_callback(progress=50.0, message=f"Streaming... ({chunk_count} chunks)")
                    except Exception:
                        pass  # Ignore callback errors
                
                # Report usage if present in chunk (some APIs send it mid-stream)
                if hasattr(chunk, "usage") and chunk.usage:
                    # Actual usage from API
                    await self._log_usage(model, chunk.usage.prompt_tokens, chunk.usage.completion_tokens, project_id=project_id)
                    usage_reported = True

                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    # Support both reasoning_content and legacy thought/reasoning attributes
                    reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "thought", None)
                    if reasoning:
                        reasoning_log += reasoning
                        await get_broadcaster().update_stream(thinking=reasoning, project_id=project_id)

                    content = getattr(delta, "content", None)
                    if content:
                        full_response += content
                        await get_broadcaster().update_stream(content=content, project_id=project_id)
        finally:
            # Ensure heartbeat task is stopped
            first_chunk_received.set()
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        # CRITICAL FIX: Report usage at END of stream if not already reported
        # Coding Plan API and some other providers only send usage in the final chunk
        if not usage_reported:
            # Estimate tokens if not provided (fallback) - use minimum of 1 to avoid 0 counts
            estimated_prompt = max(1, self.estimate_tokens(messages))
            estimated_completion = max(1, self.estimate_tokens([{"role": "assistant", "content": full_response}]))
            await self._log_usage(model, estimated_prompt, estimated_completion, project_id=project_id)

        if include_reasoning and reasoning_log:
            output = f"<thought>\n{reasoning_log.strip()}\n</thought>\n\n{full_response.strip()}".strip()
        else:
            output = (full_response or reasoning_log).strip() or "Error: Empty stream response."
            
        return await self._prepend_warnings(output)

    async def _standard_completion(self, model, messages, temp, max_t, timeout, include_reasoning, project_id="default"):
        client_to_use = self.get_client_for_model(model)
        response = await client_to_use.chat.completions.create(
            model=model, messages=messages, temperature=temp, max_tokens=max_t,
            timeout=timeout, stream=False,
        )

        project_id = project_id
        if hasattr(response, "usage") and response.usage:
            await self._log_usage(model, response.usage.prompt_tokens, response.usage.completion_tokens, project_id=project_id)
        else:
            # FALLBACK: Coding Plan and some APIs don't return usage - estimate it
            estimated_prompt = max(1, self.estimate_tokens(messages))
            estimated_completion = max(1, self.estimate_tokens([{"role": "assistant", "content": getattr(response.choices[0].message, "content", "") or ""}]))
            await self._log_usage(model, estimated_prompt, estimated_completion, project_id=project_id)

        if not response.choices:
            return "Error: Empty response."

        msg = response.choices[0].message
        content = getattr(msg, "content", "") or ""
        reasoning = getattr(msg, "reasoning_content", "") or getattr(msg, "thought", "") or ""

        if include_reasoning and reasoning:
            output = f"<thought>\n{reasoning.strip()}\n</thought>\n\n{content.strip()}".strip()
        else:
            output = (content or reasoning).strip() or "Error: Empty content."
            
        return await self._prepend_warnings(output)

    async def _prepend_warnings(self, content: str) -> str:
        """Attaches active warnings to the response string and clears the warning queue."""
        if hasattr(self, "active_warnings") and self.active_warnings:
            warnings_str = "\n".join(self.active_warnings)
            self.active_warnings.clear()  # Drain the queue
            delimiter = "-" * 40
            return f"{warnings_str}\n{delimiter}\n\n{content}"
        return content

    async def _log_usage(self, model, prompt, completion, project_id: str = "default"):
        """Standardized billing and HUD telemetry logging.
        
        Args:
            prompt: Prompt tokens (int) or prompt text (str) - will be estimated if string
            completion: Completion tokens (int) or completion text (str) - will be estimated if string
        """
        # DEBUG: Log input types
        logger.info(f"[DEBUG _log_usage] model={model}, prompt type={type(prompt)}, completion type={type(completion)}")
        
        # Convert to token counts (handle both int tokens and str content)
        prompt_tokens = prompt if isinstance(prompt, int) else self.estimate_tokens([{"role": "user", "content": str(prompt)}]) if prompt else 0
        completion_tokens = completion if isinstance(completion, int) else self.estimate_tokens([{"role": "assistant", "content": str(completion)}]) if completion else 0
        
        # DEBUG: Log converted types
        logger.info(f"[DEBUG _log_usage] prompt_tokens={prompt_tokens} (type={type(prompt_tokens)}), completion_tokens={completion_tokens} (type={type(completion_tokens)})")
        
        if model not in self.session_usage:
            self.session_usage[model] = {"prompt": 0, "completion": 0}
        self.session_usage[model]["prompt"] += prompt_tokens
        self.session_usage[model]["completion"] += completion_tokens

        project_name = project_id
        try:
            billing_tracker.log_usage(project_name, model, prompt_tokens, completion_tokens)
            await get_broadcaster().report_usage(model, prompt_tokens, completion_tokens, project_id=project_id)
        except Exception as e:
            logger.error(f"Telemetry logging failed: {e}")

    async def _handle_error(self, e, timeout):
        """Centralized error translation."""
        if isinstance(e, APITimeoutError):
            return f"Error: API Timeout (>{timeout}s). The request took too long to complete."
        if isinstance(e, APIError):
            # Some APIError subclasses may not have status_code
            status_code = getattr(e, 'status_code', 'unknown')
            message = getattr(e, 'message', str(e))
            return f"Error: API Error ({status_code}): {message}"
        if "model_not_found" in str(e).lower() or "not found" in str(e).lower():
            return f"Error: Model not found. {str(e)}"
        logger.exception("Unexpected Completion Error")
        return f"Error: {str(e)}"

    async def heal_registry(self) -> str:
        """Stub for registry healing - to be inherited or shared."""
        # This will be fully implemented or called from api.py DashScopeClient
        return "Healing triggered (stub)."
