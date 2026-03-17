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
from .specter.identity import get_current_project_id
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
    ) -> str:
        """Generate common chat completion with financial circuit breakers and retries."""
        request_timeout = timeout or self.default_timeout
        force_max_tokens = max_tokens or self.max_output_tokens

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
            
            project_id = get_current_project_id()
            logger.info(f"Completion Request | Model: {target_model} | Attempt: {attempt+1} | Project: {project_id}")

            # Telemetry: Update total tokens in HUD with estimated plan tokens
            await get_broadcaster().broadcast_state({
                "active_model": target_model,
                "request_tokens": {"prompt": estimated_input, "completion": 0}
            }, project_id=project_id)

            # Auto-detect reasoning models
            is_reasoning_model = "qwq" in target_model.lower() or task_type == "analyst"
            use_streaming = bool(progress_callback) or is_reasoning_model
            extra_body = {"enable_thinking": True} if is_reasoning_model else {}

            try:
                if use_streaming:
                    return await self._stream_completion(
                        target_model, messages, temperature, force_max_tokens, 
                        request_timeout, extra_body, include_reasoning, project_id=project_id
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

    async def _stream_completion(self, model, messages, temp, max_t, timeout, extra, include_reasoning, project_id="default"):
        client_to_use = self.get_client_for_model(model)
        response = await client_to_use.chat.completions.create(
            model=model, messages=messages, temperature=temp, max_tokens=max_t,
            timeout=timeout, stream=True, stream_options={"include_usage": True},
            extra_body=extra if extra else None,
        )

        full_response = ""
        reasoning_log = ""
        
        async for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                # Actual usage from API
                await self._log_usage(model, chunk.usage.prompt_tokens, chunk.usage.completion_tokens, project_id=project_id)

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

        if hasattr(response, "usage") and response.usage:
            await self._log_usage(model, response.usage.prompt_tokens, response.usage.completion_tokens, project_id=project_id)

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
        """Standardized billing and HUD telemetry logging."""
        if model not in self.session_usage:
            self.session_usage[model] = {"prompt": 0, "completion": 0}
        self.session_usage[model]["prompt"] += prompt
        self.session_usage[model]["completion"] += completion

        project_name = project_id
        try:
            billing_tracker.log_usage(project_name, model, prompt, completion)
            await get_broadcaster().report_usage(model, prompt, completion, project_id=project_id)
        except Exception as e:
            logger.error(f"Telemetry logging failed: {e}")

    async def _handle_error(self, e, timeout):
        """Centralized error translation."""
        if isinstance(e, APIError):
            return f"Error: API Error ({e.status_code}): {e.message}"
        if "model_not_found" in str(e).lower() or "not found" in str(e).lower():
            return f"Error: Model not found. {str(e)}"
        logger.exception("Unexpected Completion Error")
        return f"Error: {str(e)}"

    async def heal_registry(self) -> str:
        """Stub for registry healing - to be inherited or shared."""
        # This will be fully implemented or called from api.py DashScopeClient
        return "Healing triggered (stub)."
