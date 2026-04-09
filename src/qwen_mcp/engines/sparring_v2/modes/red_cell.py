"""
Sparring Engine v2 - Red Cell Mode Executor

Execute Red Cell critique (adversarial audit).
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
from qwen_mcp.engines.sparring_v2.helpers import validate_session, get_model
from qwen_mcp.prompts.sparring import RED_CELL_PROMPT, get_word_limit_instruction
from qwen_mcp.sanitizer import ContentValidator

logger = logging.getLogger(__name__)


class RedCellExecutor(ModeExecutor):
    """Execute Red Cell critique."""
    
    async def execute(
        self,
        session_id: Optional[str] = None,
        ctx: Optional[Context] = None,
        word_limit: Optional[int] = None,
    ) -> SparringResponse:
        """
        Execute Red Cell critique.
        
        Args:
            session_id: Session ID from discovery step
            ctx: MCP context for progress reporting
            word_limit: Optional word limit for responses
            
        Returns:
            SparringResponse with critique results
        """
        logger.info(f"Executing Red Cell mode for session: {session_id}")
        
        # Validate session
        loaded_session = self.session_store.load(session_id) if session_id else None
        session, error_response = validate_session(loaded_session, session_id, "red")
        if error_response:
            return error_response
        
        # Checkpoint: Mark step as in-progress
        session.current_step = "red"
        self.session_store.save(session)
        
        await self._report_progress(ctx, 0.0, f"[Red Cell] {session.roles.get('red_role', 'Red')} auditing...")
        
        # Build messages with optional word limit
        red_prompt = RED_CELL_PROMPT
        if word_limit:
            red_prompt += get_word_limit_instruction(word_limit)
        
        # Multi-turn support: Include conversation history from SessionStore
        conversation_history = self.session_store.get_messages_for_api(session_id) if session_id else []
        
        red_messages = [
            {"role": "system", "content": f"Jesteś {session.roles.get('red_role', 'Red Cell')}. Profil: {session.roles.get('red_profile', '')}\n\nZADANIE:\n{red_prompt}"},
            {"role": "user", "content": f"Topic: {session.topic}\n\nContext: {session.context}"},
        ]
        
        # Append conversation history for multi-turn context
        red_messages.extend(conversation_history)
        
        # Execute API call
        # Determine mode key based on word_limit (full mode uses word_limit, pro mode doesn't)
        mode_key = "full" if word_limit else "pro"
        red_model = get_model(session, "red_model")
        logger.debug(f"[Red Cell] red_model: {red_model}, roles type: {type(session.roles)}, roles: {session.roles}")
        red_critique = await self.client.generate_completion(
            messages=red_messages,
            temperature=0.8,
            task_type="audit",
            timeout=TIMEOUTS["red_cell"],
            max_tokens=get_max_tokens_for_step(mode_key, "red"),
            thinking_budget=get_thinking_tokens_for_mode("sparring2" if word_limit else "sparring3", "red"),
            complexity="critical",
            tags=["sparring", "red-cell"],
            include_reasoning=True,
            model_override=red_model,
            enable_thinking=True,
        )
        red_critique = ContentValidator.validate_response(red_critique)
        
        # Update session
        self.session_store.update_step(
            session_id, "red",
            {"critique": red_critique, "raw": red_critique},
            next_step="blue",
        )
        
        return SparringResponse.success(
            session_id=session_id,
            step="red",
            next_step="blue",
            next_command=f"sparring(session_id='{session_id}', mode='blue')",
            result={"critique": red_critique},
            message="Red Cell critique complete",
        )
    
    # _report_progress inherited from ModeExecutor base class (with WebSocket broadcast)
