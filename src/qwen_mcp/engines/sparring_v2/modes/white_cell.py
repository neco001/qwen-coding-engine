"""
Sparring Engine v2 - White Cell Mode Executor

Execute White Cell synthesis with optional regeneration loop.
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
from qwen_mcp.prompts.sparring import WHITE_CELL_PROMPT, BLUE_CELL_PROMPT, get_word_limit_instruction
from qwen_mcp.sanitizer import ContentValidator

logger = logging.getLogger(__name__)


class WhiteCellExecutor(ModeExecutor):
    """Execute White Cell synthesis with optional regeneration loop."""
    
    async def execute(
        self,
        session_id: Optional[str] = None,
        ctx: Optional[Context] = None,
        allow_regeneration: bool = True,
        word_limit: Optional[int] = None,
    ) -> SparringResponse:
        """
        Execute White Cell synthesis.
        
        Args:
            session_id: Session ID from discovery step
            ctx: MCP context for progress reporting
            allow_regeneration: Whether to allow regeneration loop
            word_limit: Optional word limit for responses
            
        Returns:
            SparringResponse with consensus and report
        """
        logger.info(f"Executing White Cell mode for session: {session_id}")
        
        # Validate session
        loaded_session = self.session_store.load(session_id) if session_id else None
        session, error_response = validate_session(loaded_session, session_id, "white")
        if error_response:
            return error_response
        
        # Get prerequisites
        red_critique = get_step_result(session, "red", "critique")
        blue_defense = get_step_result(session, "blue", "defense")
        
        if not red_critique or not blue_defense:
            return SparringResponse.error(
                message="Red/Blue results not found. Complete previous steps first.",
                error="Missing prerequisites",
                session_id=session_id,
            )
        
        # Checkpoint: Mark step as in-progress
        session.current_step = "white"
        self.session_store.save(session)
        
        # Regeneration loop (disabled in full mode to avoid MCP 300s timeout)
        max_loops = 1 if not allow_regeneration else 2
        loop_count = session.loop_count
        white_consensus = ""
        
        # DEBUG: Log loop_count type
        logger.debug(f"[White Cell] loop_count type: {type(loop_count)}, value: {loop_count}")
        logger.debug(f"[White Cell] max_loops type: {type(max_loops)}, value: {max_loops}")
        
        # Build prompt with optional word limit
        white_prompt = WHITE_CELL_PROMPT
        if word_limit:
            white_prompt += get_word_limit_instruction(word_limit)
        
        # Multi-turn support: Include conversation history from SessionStore
        conversation_history = self.session_store.get_messages_for_api(session_id) if session_id else []
        
        while loop_count < max_loops:
            loop_count += 1
            await self._report_progress(ctx, 0.0, f"[White Cell] {session.roles.get('white_role', 'White')} synthesizing (loop {loop_count})...")
            
            white_messages = [
                {"role": "system", "content": f"Jesteś {session.roles.get('white_role', 'White Cell')}. Profil: {session.roles.get('white_profile', '')}\n\nZADANIE:\n{white_prompt}"},
                {"role": "user", "content": f"Topic: {session.topic}\n\nContext: {session.context}\n\nRed Audit:\n{red_critique}\n\nBlue Defense:\n{blue_defense}"},
            ]
            
            # Append conversation history for multi-turn context
            white_messages.extend(conversation_history)
            
            # Execute API call
            # Determine mode key based on word_limit (full mode uses word_limit, pro mode doesn't)
            mode_key = "full" if word_limit else "pro"
            white_model = get_model(session, "white_model")
            white_consensus = await self.client.generate_completion(
                messages=white_messages,
                temperature=0.1,
                task_type="audit",
                timeout=TIMEOUTS["white_cell"],
                max_tokens=get_max_tokens_for_step(mode_key, "white"),
                thinking_budget=get_thinking_tokens_for_mode("sparring2" if word_limit else "sparring3", "white"),
                complexity="critical",
                tags=["sparring", "white-cell", f"loop-{loop_count}"],
                include_reasoning=True,
                model_override=white_model,
                enable_thinking=True,
            )
            white_consensus = ContentValidator.validate_response(white_consensus)
            
            # Check for regeneration request
            if "[REGENERATE" not in white_consensus or loop_count >= max_loops:
                if "[REGENERATE" in white_consensus:
                    white_consensus = white_consensus.split("]", 1)[-1].strip()
                break
            
            # If regeneration requested, get new blue defense
            await self._report_progress(ctx, 50.0, "[White Cell] Requesting Blue Cell regeneration...")
            
            regen_reason = white_consensus.split("[REGENERATE:", 1)[1].split("]", 1)[0].strip() if "[REGENERATE:" in white_consensus else "Improvement needed"
            
            blue_messages = [
                {"role": "system", "content": f"Jesteś {session.roles.get('blue_role', 'Blue Cell')}. Profil: {session.roles.get('blue_profile', '')}\n\nZADANIE:\n{BLUE_CELL_PROMPT}\n\nUWAGA: Poprzednia próba odrzucona: {regen_reason}"},
                {"role": "user", "content": f"Topic: {session.topic}\n\nContext: {session.context}\n\nRed Critique:\n{red_critique}"},
            ]
            
            # Execute regeneration API call
            # Determine mode key based on word_limit (full mode uses word_limit, pro mode doesn't)
            mode_key = "full" if word_limit else "pro"
            regen_blue_model = get_model(session, "blue_model")
            blue_defense = await self.client.generate_completion(
                messages=blue_messages,
                temperature=0.5,
                task_type="audit",
                timeout=TIMEOUTS["blue_cell"],
                max_tokens=get_max_tokens_for_step(mode_key, "blue"),
                thinking_budget=get_thinking_tokens_for_mode("sparring2" if word_limit else "sparring3", "blue"),
                complexity="high",
                tags=["sparring", "blue-cell", "regen"],
                include_reasoning=True,
                model_override=regen_blue_model,
                enable_thinking=True,
            )
            blue_defense = ContentValidator.validate_response(blue_defense)
            
            # Update session with new blue defense
            session.results["blue"] = {"defense": blue_defense, "raw": blue_defense}
            self.session_store.save(session)
        
        # Update session
        session.loop_count = loop_count
        self.session_store.update_step(
            session_id, "white",
            {"consensus": white_consensus, "raw": white_consensus, "loops": loop_count},
        )
        
        # Assemble final report
        report = self._assemble_report(session, red_critique, blue_defense, white_consensus, loop_count)
        
        return SparringResponse.success(
            session_id=session_id,
            step="white",
            next_step=None,
            next_command=None,
            result={"consensus": white_consensus, "report": report, "loops": loop_count},
            message=f"White Cell synthesis complete ({loop_count} loop{'s' if loop_count > 1 else ''})",
        )
    
    def _assemble_report(
        self,
        session,
        red: str,
        blue: str,
        white: str,
        loops: int,
    ) -> str:
        """Assemble final war game report."""
        report = f"# 🛡️ War Game Report: {session.topic}\n\n"
        report += f"> **CONFIDENTIAL: STRATEGIC DRAFT ONLY. NOT FOR EXTERNAL DISTRIBUTION.**\n\n"
        report += f"> **Session ID:** `{session.session_id}`\n\n"
        report += f"> **Selected Roles:** {session.roles.get('red_role', 'Red')}, {session.roles.get('blue_role', 'Blue')}, {session.roles.get('white_role', 'White')}\n\n"
        report += f"## 🥊 Turn 2: {session.roles.get('red_role', 'Red')}\n\n{self._format_output(red, session.roles.get('red_role', 'Red'))}\n\n---\n\n"
        report += f"## 🛡️ Turn 3: {session.roles.get('blue_role', 'Blue')}\n\n{self._format_output(blue, session.roles.get('blue_role', 'Blue'))}\n\n---\n\n"
        report += f"## ⚖️ Turn 4: {session.roles.get('white_role', 'White')}\n\n{self._format_output(white, session.roles.get('white_role', 'White'))}\n\n"
        # Ensure loops is an int (handle JSON deserialization edge cases)
        loops_int = int(loops) if not isinstance(loops, int) else loops
        if loops_int > 1:
            report += f"\n\n*(Note: This report underwent {loops_int} optimization cycles)*"
        return report
    
    def _format_output(self, raw: str, label: str) -> str:
        """Format output with reasoning hidden in details."""
        if not isinstance(raw, str):
            raw = str(raw) if raw is not None else ""
        
        if "<thought>" in raw:
            parts = raw.split("</thought>")
            thought = parts[0].replace("<thought>", "").strip()
            content = parts[1].strip() if len(parts) > 1 else ""
            return f"<details>\n<summary>🧠 Proces Myślowy ({label})</summary>\n\n{thought}\n</details>\n\n{content}"
        return ContentValidator.validate_response(raw)
    
    # _report_progress inherited from ModeExecutor base class (with WebSocket broadcast)
