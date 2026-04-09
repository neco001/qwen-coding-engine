"""
Sparring Engine v2 - Discovery Mode Executor

Define roles and create session for sparring.
"""

import logging
from typing import Optional, Dict, Any
from mcp.server.fastmcp import Context

from qwen_mcp.engines.sparring_v2.interfaces import ModeExecutor
from qwen_mcp.engines.sparring_v2.models import SparringResponse
from qwen_mcp.engines.sparring_v2.config import (
    TIMEOUTS,
    get_max_tokens_for_step,
    get_thinking_tokens_for_mode,
)
from qwen_mcp.engines.sparring_v2.helpers import get_model
from qwen_mcp.prompts.sparring import get_discovery_prompt, get_word_limit_instruction
from qwen_mcp.tools import extract_json_from_text
from qwen_mcp.registry import ModelRegistry
from qwen_mcp.base import get_billing_mode

logger = logging.getLogger(__name__)


class DiscoveryExecutor(ModeExecutor):
    """Execute discovery mode: Define roles and create session."""
    
    async def execute(
        self,
        topic: str,
        context: str = "",
        ctx: Optional[Context] = None,
        word_limit: Optional[int] = None,
    ) -> SparringResponse:
        """
        Execute discovery mode.
        
        Args:
            topic: Topic for the sparring
            context: Additional context
            ctx: MCP context for progress reporting
            word_limit: Optional word limit for responses
            
        Returns:
            SparringResponse with discovered roles and session info
        """
        logger.info(f"Executing discovery mode for topic: {topic}")
        
        await self._report_progress(ctx, 0.0, "[Discovery] Assembling Expert Bench...")
        
        # Get billing mode for model selection
        billing_mode = get_billing_mode()
        discovery_prompt = get_discovery_prompt(billing_mode)
        
        # Inject word limit for full mode (shorter responses)
        if word_limit:
            discovery_prompt += get_word_limit_instruction(word_limit)
        
        discovery_messages = [
            {"role": "system", "content": discovery_prompt},
            {"role": "user", "content": f"Topic: {topic}\n\nContext: {context}"},
        ]
        
        # Use fast model for discovery - it's just JSON extraction
        # Determine mode key based on word_limit (full mode uses word_limit, pro mode doesn't)
        mode_key = "full" if word_limit else "pro"
        discovery_raw = await self.client.generate_completion(
            messages=discovery_messages,
            temperature=0.0,
            task_type="scout",  # Use scout (kimi-k2.5) for fast JSON extraction
            timeout=TIMEOUTS["discovery"],
            max_tokens=get_max_tokens_for_step(mode_key, "discovery"),
            thinking_budget=get_thinking_tokens_for_mode("sparring2" if word_limit else "sparring3", "discovery"),
            complexity="low",
            tags=["sparring", "discovery"],
        )
        
        # Parse roles and models
        try:
            parsed = extract_json_from_text(discovery_raw)
            required_keys = ["red_role", "red_profile", "blue_role", "blue_profile",
                           "white_role", "white_profile"]
            if not parsed or not all(k in parsed for k in required_keys):
                raise ValueError("Incomplete role discovery")
            
            # Extract roles
            roles = {k: v for k, v in parsed.items() if not k.endswith("_model")}
            
            # Extract models with validation against billing mode
            raw_models = {
                "red_model": parsed.get("red_model", "glm-5"),
                "blue_model": parsed.get("blue_model", "qwen3.5-plus"),
                "white_model": parsed.get("white_model", "qwen3.5-plus"),
            }
            
            # Validate each model against billing mode
            models: Dict[str, str] = {}
            for role_key, model_id in raw_models.items():
                is_valid, result = ModelRegistry.validate_override(model_id, billing_mode)
                if is_valid:
                    models[role_key] = model_id
                else:
                    # Use fallback model
                    task_type = "analyst" if role_key == "red_model" else "strategist"
                    fallback = ModelRegistry.get_fallback_model(billing_mode, task_type)
                    models[role_key] = fallback
                    logger.warning(f"[Discovery] {result}. Using fallback: {fallback}")
            
        except Exception:
            # Fallback to default roles and models
            roles = {
                "red_role": "Red Cell (Adversarial Audit)",
                "red_profile": "Cyniczny audytor metod i niuansów persony",
                "blue_role": "Blue Cell (Strategic Defense)",
                "blue_profile": "Adwokat użytkownika i autentyczności tonu",
                "white_role": "White Cell (Final Consensus)",
                "white_profile": "Chief of Staff dbający o logiczną spójność i ROI",
            }
            # Use validated fallbacks
            models = {
                "red_model": ModelRegistry.get_fallback_model(billing_mode, "analyst"),
                "blue_model": ModelRegistry.get_fallback_model(billing_mode, "strategist"),
                "white_model": ModelRegistry.get_fallback_model(billing_mode, "strategist"),
            }
            logger.warning("Using default roles and models due to discovery parse failure")
        
        # Create session
        session = self.session_store.create_session(topic=topic, context=context)
        session.roles = roles
        session.models = models
        self.session_store.save(session)
        
        # CRITICAL: Log session_id for debugging
        logger.info(f"DiscoveryExecutor created session: {session.session_id!r}")
        
        return SparringResponse(
            success=True,
            session_id=session.session_id,
            step_completed="discovery",
            next_step="red",
            next_command=f"sparring(session_id='{session.session_id}', mode='red')",
            result={"roles": roles, "models": models},
            message="Discovery complete. Session created.",
        )
    
    # _report_progress inherited from ModeExecutor base class (with WebSocket broadcast)
