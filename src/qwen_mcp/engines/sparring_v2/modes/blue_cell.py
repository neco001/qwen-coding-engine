"""
Sparring Engine v2 - Blue Cell Mode Executor

Execute Blue Cell defense (strategic response to Red Cell critique).
"""

import logging
from typing import Optional
from mcp.server.fastmcp import Context

from qwen_mcp.engines.sparring_v2.interfaces import ModeExecutor
from qwen_mcp.engines.sparring_v2.models import SparringResponse
from qwen_mcp.engines.sparring_v2.config import (
    TIMEOUTS,
    get_max_tokens_for_step,
    get_thinking_tokens_for_mode,
)
from qwen_mcp.engines.sparring_v2.helpers import validate_session, get_model, get_step_result
from qwen_mcp.prompts.sparring import BLUE_CELL_PROMPT, get_word_limit_instruction
from qwen_mcp.sanitizer import ContentValidator

logger = logging.getLogger(__name__)


class BlueCellExecutor(ModeExecutor):
    """Execute Blue Cell defense."""
    
    async def execute(
        self,
        session_id: Optional[str] = None,
        ctx: Optional[Context] = None,
        word_limit: Optional[int] = None,
    ) -> SparringResponse:
        """
        Execute Blue Cell defense.
        
        Args:
            session_id: Session ID from discovery step
            ctx: MCP context for progress reporting
            word_limit: Optional word limit for responses
            
        Returns:
            SparringResponse with defense results
        """
        logger.info(f"Executing Blue Cell mode for session: {session_id}")
        
        # Validate session
        loaded_session = self.session_store.load(session_id) if session_id else None
        session, error_response = validate_session(loaded_session, session_id, "blue")
        if error_response:
            return error_response
        
        # Get red critique prerequisite
        red_critique = get_step_result(session, "red", "critique")
        if not red_critique:
            return SparringResponse.error(
                message="Red Cell critique not found. Run discovery and red first.",
                error="Missing red critique",
                session_id=session_id,
            )
        
        # Checkpoint: Mark step as in-progress
        session.current_step = "blue"
        self.session_store.save(session)
        
        await self._report_progress(ctx, 0.0, f"[Blue Cell] {session.roles.get('blue_role', 'Blue')} defending...")
        
        # Build messages with optional word limit
        blue_prompt = BLUE_CELL_PROMPT
        if word_limit:
            blue_prompt += get_word_limit_instruction(word_limit)
        
        # Multi-turn support: Include conversation history from SessionStore
        conversation_history = self.session_store.get_messages_for_api(session_id) if session_id else []
        
        blue_messages = [
            {"role": "system", "content": f"Jesteś {session.roles.get('blue_role', 'Blue Cell')}. Profil: {session.roles.get('blue_profile', '')}\n\nZADANIE:\n{blue_prompt}"},
            {"role": "user", "content": f"Topic: {session.topic}\n\nContext: {session.context}\n\nRed Critique:\n{red_critique}"},
        ]
        
        # Append conversation history for multi-turn context
        blue_messages.extend(conversation_history)
        
        # Execute API call
        # Determine mode key based on word_limit (full mode uses word_limit, pro mode doesn't)
        mode_key = "full" if word_limit else "pro"
        blue_model = get_model(session, "blue_model")
        blue_defense = await self.client.generate_completion(
            messages=blue_messages,
            temperature=0.5,
            task_type="audit",
            timeout=TIMEOUTS["blue_cell"],
            max_tokens=get_max_tokens_for_step(mode_key, "blue"),
            thinking_budget=get_thinking_tokens_for_mode("sparring2" if word_limit else "sparring3", "blue"),
            complexity="critical",
            tags=["sparring", "blue-cell"],
            include_reasoning=True,
            model_override=blue_model,
            enable_thinking=True,
        )
        blue_defense = ContentValidator.validate_response(blue_defense)
        
        # Update session
        self.session_store.update_step(
            session_id, "blue",
            {"defense": blue_defense, "raw": blue_defense},
            next_step="white",
        )
        
        return SparringResponse.success(
            session_id=session_id,
            step="blue",
            next_step="white",
            next_command=f"sparring(session_id='{session_id}', mode='white')",
            result={"defense": blue_defense},
            message="Blue Cell defense complete",
        )
    
    # _report_progress inherited from ModeExecutor base class (with WebSocket broadcast)
