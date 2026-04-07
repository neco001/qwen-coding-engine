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
        thinking_budget: Optional[int] = None,
        project_id: str = "default",
    ) -> str:
        """Generate common chat completion with financial circuit breakers and retries.
        
        Args:
            enable_thinking: Override thinking mode. None = auto-detect for reasoning models.
            thinking_budget: Limit thinking tokens for deep-thinking models (default: 2048).
            project_id: Project/session ID for telemetry isolation (format: {instance}_{source}_{hash}).
        """
        request_timeout = timeout or self.default_timeout
        
        # TokenScout: Dynamic max_tokens based on prompt analysis
        # Replaces hardcoded complexity-based limits
        from qwen_mcp.engines.token_scout import TokenScout, SAFETY_MAX_TOKENS
        
        scout = TokenScout()
        
        if max_tokens:
            # Explicit override - use it
            force_max_tokens = max_tokens
        else:
            # Extract prompt from messages for estimation
            prompt_text = ""
            for msg in messages:
                if msg.get("role") == "user" and "content" in msg:
                    prompt_text += msg["content"] + "\n"
            
            # Use TokenScout to estimate output tokens
            estimation = scout.estimate_output_tokens(prompt_text)
            force_max_tokens = scout.get_max_tokens(estimation["estimated_tokens"])
            
            logger.info(f"TokenScout estimate: {estimation['estimated_tokens']} tokens \u2192 max_tokens={force_max_tokens} (confidence: {estimation['confidence']})")

        # Check circuit breaker
        try:
            self.check_financial_circuit_breaker(messages)
        except ValueError as e:
            return f"\u274c {str(e)} Please truncate your files or context."

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
                if thinking_budget:
                    thinking_budget = thinking_budget  # Use provided value
                elif complexity == "critical":
                    thinking_budget = 4096
                elif complexity == "high":
                    thinking_budget = 2048
                else:  # low, medium, auto
                    thinking_budget = 1024
                extra_body["thinking_budget"] = thinking_budget
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
                error_str = str(e).lower()
                is_model_error = "model_not_found" in error_str or "not found" in error_str
                is_token_limit_error = "invalid_parameter" in error_str or "max_tokens" in error_str or "65536" in error_str
                
                # Retry on model_not_found (refresh registry)
                if is_model_error and attempt == 0:
                    logger.warning(f"Model {target_model} rejected. Refreshing registry...")
                    await self.heal_registry()
                    continue
                
                # Retry ONCE on token limit errors (HTTP 400)
                if is_token_limit_error and attempt == 0:
                    logger.warning(
                        f"Token limit error detected (attempt {attempt+1}): {e}. "
                        f"Retrying with reduced max_tokens ({force_max_tokens} \u2192 {min(force_max_tokens, 30000)})..."
                    )
                    # Reduce max_tokens for retry
                    force_max_tokens = min(force_max_tokens, 30000)
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
            # Wait only 3 seconds before starting heartbeats (MCP default timeout is ~30s)
            await asyncio.sleep(3)
            iteration = 0
            while not first_chunk_received.is_set():
                if progress_callback:
                    try:
                        iteration += 1
                        # Send MCP progress notification (for protocol compliance)
                        await progress_callback(progress=15.0, message=f"Model is thinking... ({iteration * 5}s)")
                        # ALSO broadcast via WebSocket for UI visibility (Roo Code HUD listens to this)
                        await get_broadcaster().broadcast_state({
                            "operation": f"Model is thinking... ({iteration * 5}s)",
                            "progress": 15.0,
                            "is_live": True
                        }, project_id=project_id)
                    except Exception as e:
                        logger.debug(f"Heartbeat broadcast error: {e}")
                await asyncio.sleep(5)  # Send heartbeat every 5 seconds

        heartbeat_task = asyncio.create_task(heartbeat_loop())

        try:
            async for chunk in response:
                if not first_chunk_received.is_set():
                    first_chunk_received.set()
                
                chunk_count += 1
                
                # CRITICAL: Call progress_callback every 10 chunks to keep MCP connection alive
                if progress_callback and chunk_count % 10 == 0:
                    try:
                        # Send MCP progress notification (for protocol compliance)
                        await progress_callback(progress=50.0, message=f"Streaming... ({chunk_count} chunks)")
                        # ALSO broadcast via WebSocket for UI visibility (Roo Code HUD listens to this)
                        await get_broadcaster().broadcast_state({
                            "operation": f"Streaming response... ({chunk_count} chunks)",
                            "progress": 50.0,
                            "is_live": True
                        }, project_id=project_id)
                    except Exception as e:
                        logger.debug(f"Streaming broadcast error: {e}")
                
                if hasattr(chunk, "usage") and chunk.usage:
                    await self._log_usage(model, chunk.usage.prompt_tokens, chunk.usage.completion_tokens, project_id=project_id)
                    usage_reported = True

                if hasattr(chunk, "choices") and chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "thought", None)
                    if reasoning:
                        reasoning_log += reasoning
                        await get_broadcaster().update_stream(thinking=reasoning, project_id=project_id)

                    content = getattr(delta, "content", None)
                    if content:
                        full_response += content
                        await get_broadcaster().update_stream(content=content, project_id=project_id)
        finally:
            first_chunk_received.set()
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        if not usage_reported:
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
        
        if not response.choices or len(response.choices) == 0:
            return "Error: Empty response choices from API."

        if hasattr(response, "usage") and response.usage:
            await self._log_usage(model, response.usage.prompt_tokens, response.usage.completion_tokens, project_id=project_id)
        else:
            estimated_prompt = max(1, self.estimate_tokens(messages))
            content_for_est = getattr(response.choices[0].message, "content", "") or ""
            estimated_completion = max(1, self.estimate_tokens([{"role": "assistant", "content": content_for_est}]))
            await self._log_usage(model, estimated_prompt, estimated_completion, project_id=project_id)

        msg = response.choices[0].message
        content = getattr(msg, "content", "") or ""
        reasoning = getattr(msg, "reasoning_content", "") or getattr(msg, "thought", "") or ""

        if include_reasoning and reasoning:
            output = f"<thought>\n{reasoning.strip()}\n</thought>\n\n{content.strip()}".strip()
        else:
            output = (content or reasoning).strip() or "Error: Empty content."
            
        return await self._prepend_warnings(output)

    async def _prepend_warnings(self, content: str) -> str:
        if hasattr(self, "active_warnings") and self.active_warnings:
            warnings_str = "\n".join(self.active_warnings)
            self.active_warnings.clear()
            delimiter = "-" * 40
            return f"{warnings_str}\n{delimiter}\n\n{content}"
        return content

    async def _log_usage(self, model, prompt, completion, project_id: str = "default"):
        logger.info(f"Testing usage log: {model} {prompt} {completion}")
        prompt_tokens = prompt if isinstance(prompt, int) else self.estimate_tokens([{"role": "user", "content": str(prompt)}]) if prompt else 0
        completion_tokens = completion if isinstance(completion, int) else self.estimate_tokens([{"role": "assistant", "content": str(completion)}]) if completion else 0
        
        if model not in self.session_usage:
            self.session_usage[model] = {"prompt": 0, "completion": 0}
        self.session_usage[model]["prompt"] += prompt_tokens
        self.session_usage[model]["completion"] += completion_tokens

        try:
            billing_tracker.log_usage(project_id, model, prompt_tokens, completion_tokens)
            await get_broadcaster().report_usage(model, prompt_tokens, completion_tokens, project_id=project_id)
        except Exception as e:
            logger.error(f"Telemetry logging failed: {e}")

    async def _handle_error(self, e, timeout):
        if isinstance(e, APITimeoutError):
            return f"Error: API Timeout (>{timeout}s). The request took too long to complete."
        if isinstance(e, APIError):
            status_code = getattr(e, 'status_code', 'unknown')
            message = getattr(e, 'message', str(e))
            return f"Error: API Error ({status_code}): {message}"
        if "model_not_found" in str(e).lower() or "not found" in str(e).lower():
            return f"Error: Model not found. {str(e)}"
        logger.exception("Unexpected Completion Error")
        return f"Error: {str(e)}"

    async def heal_registry(self) -> str:
        return "Healing triggered (stub)."
