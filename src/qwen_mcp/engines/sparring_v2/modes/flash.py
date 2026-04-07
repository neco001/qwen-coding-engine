"""
Sparring Engine v2 - Flash Mode Executor

Fast 2-step analysis: Analyst → Drafter (single call, no checkpoint).
"""

import logging
from typing import Optional
from mcp.server.fastmcp import Context

from qwen_mcp.engines.sparring_v2.interfaces import ModeExecutor
from qwen_mcp.engines.sparring_v2.models import SparringResponse
from qwen_mcp.engines.sparring_v2.config import (
    TIMEOUTS,
    MAX_TOKENS_CONFIG,
    MAX_THINKING_TOKENS_CONFIG,
    get_thinking_tokens_for_mode,
)
from qwen_mcp.prompts.sparring import FLASH_ANALYST_PROMPT, FLASH_DRAFTER_PROMPT

logger = logging.getLogger(__name__)


class FlashExecutor(ModeExecutor):
    """Execute flash mode: Analyst → Drafter (single call, no checkpoint)."""
    
    async def execute(
        self,
        topic: str,
        context: str = "",
        ctx: Optional[Context] = None,
    ) -> SparringResponse:
        """
        Execute flash mode analysis.
        
        Args:
            topic: Topic for the sparring
            context: Additional context
            ctx: MCP context for progress reporting
            
        Returns:
            SparringResponse with analysis results
        """
        logger.info(f"Executing flash mode for topic: {topic}")
        
        # Immediate heartbeat to prevent client timeout during model initialization
        await self._report_progress(ctx, 0.0, "[Flash] Initializing Analysis...")
        
        # Step 1: Analyst
        analyst_messages = [
            {"role": "system", "content": FLASH_ANALYST_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\n\nContext:\n{context}"},
        ]
        
        analysis = await self.client.generate_completion(
            messages=analyst_messages,
            temperature=0.7,
            task_type="audit",
            timeout=TIMEOUTS["flash_analyst"],
            max_tokens=MAX_TOKENS_CONFIG["flash"]["analyst"],
            thinking_budget=get_thinking_tokens_for_mode("sparring1"),
            complexity="high",
            tags=["sparring", "flash-analyst"],
            progress_callback=ctx.report_progress if ctx else None,
        )
        
        await self._report_progress(ctx, 50.0, "[Flash] Turn 2: Drafting Strategy...")
        
        # Step 2: Drafter
        drafter_messages = [
            {"role": "system", "content": FLASH_DRAFTER_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\n\nContext: {context}\n\nAnalysis:\n{analysis}"},
        ]
        
        final_strategy = await self.client.generate_completion(
            messages=drafter_messages,
            temperature=0.1,
            task_type="strategist",
            timeout=TIMEOUTS["flash_drafter"],
            max_tokens=MAX_TOKENS_CONFIG["flash"]["drafter"],
            thinking_budget=get_thinking_tokens_for_mode("sparring1"),
            complexity="medium",
            tags=["sparring", "flash-drafter"],
            include_reasoning=False,
            progress_callback=ctx.report_progress if ctx else None,
        )
        
        # Format output
        output = self._format_output(final_strategy, "Flash Analysis")
        
        return SparringResponse(
            success=True,
            session_id=None,
            step_completed="flash",
            next_step=None,
            next_command=None,
            result={"strategy": output},
            message="Flash analysis complete",
        )
    
    # _report_progress inherited from ModeExecutor base class (with WebSocket broadcast)
    def _format_output(self, raw: str, label: str) -> str:
        """Format output with reasoning hidden in details."""
        from qwen_mcp.sanitizer import ContentValidator
        
        if not isinstance(raw, str):
            raw = str(raw) if raw is not None else ""
        
        if "<thought>" in raw:
            parts = raw.split("</thought>")
            thought = parts[0].replace("<thought>", "").strip()
            content = parts[1].strip() if len(parts) > 1 else ""
            return f"<details>\n<summary>🧠 Proces Myślowy ({label})</summary>\n\n{thought}\n</details>\n\n{content}"
        return ContentValidator.validate_response(raw)
