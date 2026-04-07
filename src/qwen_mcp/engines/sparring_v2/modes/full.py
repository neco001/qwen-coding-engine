"""
Sparring Engine v2 - Full Mode Executor

Execute complete sparring session in one call: discovery→red→blue→white.
Uses shorter word limits to ensure completion within MCP timeout.
"""

import logging
from typing import Optional
from mcp.server.fastmcp import Context

from qwen_mcp.engines.sparring_v2.interfaces import ModeExecutor
from qwen_mcp.engines.sparring_v2.models import SparringResponse
from qwen_mcp.prompts.sparring import WORD_LIMITS

logger = logging.getLogger(__name__)


class FullExecutor(ModeExecutor):
    """Execute complete sparring session in one call."""
    
    async def execute(
        self,
        topic: str,
        context: str = "",
        ctx: Optional[Context] = None,
    ) -> SparringResponse:
        """
        Execute full sparring session.
        
        Args:
            topic: Topic for the sparring
            context: Additional context
            ctx: MCP context for progress reporting
            
        Returns:
            SparringResponse with full session results
        """
        logger.info(f"Executing full mode for topic: {topic}")
        session_id = None
        
        try:
            # Import mode executors
            from qwen_mcp.engines.sparring_v2.modes.discovery import DiscoveryExecutor
            from qwen_mcp.engines.sparring_v2.modes.red_cell import RedCellExecutor
            from qwen_mcp.engines.sparring_v2.modes.blue_cell import BlueCellExecutor
            from qwen_mcp.engines.sparring_v2.modes.white_cell import WhiteCellExecutor
            
            # Create executors sharing the same client and session_store
            discovery_executor = DiscoveryExecutor(self.client, self.session_store)
            red_executor = RedCellExecutor(self.client, self.session_store)
            blue_executor = BlueCellExecutor(self.client, self.session_store)
            white_executor = WhiteCellExecutor(self.client, self.session_store)
            
            # Krok 1: Discovery (25%) - 100 words limit (JSON only)
            await self._report_progress(ctx, 25.0, "[Full] 1/4: Discovery - Defining roles...")
            discovery_result = await discovery_executor.execute(
                topic=topic,
                context=context,
                ctx=ctx,
                word_limit=WORD_LIMITS["full_discovery"],
            )
            session_id = discovery_result.session_id
            
            if not session_id:
                return SparringResponse(
                    success=False,
                    session_id=None,
                    step_completed="full",
                    next_step=None,
                    next_command=None,
                    result=None,
                    message="Full mode failed at discovery step",
                    error=discovery_result.error or "No session_id returned",
                )
            
            # Krok 2: Red Cell (50%) - 150 words limit
            await self._report_progress(ctx, 50.0, "[Full] 2/4: Red Cell - Auditing...")
            red_result = await red_executor.execute(
                session_id=session_id,
                ctx=ctx,
                word_limit=WORD_LIMITS["full_red"],
            )
            
            if not red_result.success:
                return SparringResponse(
                    success=False,
                    session_id=session_id,
                    step_completed="full",
                    next_step="red",
                    next_command=f'qwen_sparring(mode="red", session_id="{session_id}")',
                    result=None,
                    message="Full mode failed at red cell step",
                    error=red_result.error,
                )
            
            # Krok 3: Blue Cell (75%) - 150 words limit
            await self._report_progress(ctx, 75.0, "[Full] 3/4: Blue Cell - Defending...")
            blue_result = await blue_executor.execute(
                session_id=session_id,
                ctx=ctx,
                word_limit=WORD_LIMITS["full_blue"],
            )
            
            if not blue_result.success:
                return SparringResponse(
                    success=False,
                    session_id=session_id,
                    step_completed="full",
                    next_step="blue",
                    next_command=f'qwen_sparring(mode="blue", session_id="{session_id}")',
                    result=None,
                    message="Full mode failed at blue cell step",
                    error=blue_result.error,
                )
            
            # Krok 4: White Cell (100%) - 200 words limit, no regeneration
            await self._report_progress(ctx, 100.0, "[Full] 4/4: White Cell - Synthesizing (no regen)...")
            white_result = await white_executor.execute(
                session_id=session_id,
                ctx=ctx,
                allow_regeneration=False,
                word_limit=WORD_LIMITS["full_white"],
            )
            
            if not white_result.success:
                return SparringResponse(
                    success=False,
                    session_id=session_id,
                    step_completed="full",
                    next_step="white",
                    next_command=f'qwen_sparring(mode="white", session_id="{session_id}")',
                    result=None,
                    message="Full mode failed at white cell step",
                    error=white_result.error,
                )
            
            # Assemble final report
            final_report = self._assemble_full_report(
                discovery_result, red_result, blue_result, white_result,
            )
            
            return SparringResponse(
                success=True,
                session_id=session_id,
                step_completed="full",
                next_step=None,
                next_command=None,
                result={"full_report": final_report},
                message="Full sparring session complete",
            )
            
        except Exception as e:
            # Handle exceptions at each step
            logger.exception(f"Full mode failed: {e}")
            if session_id is None:
                # Failed at discovery
                return SparringResponse(
                    success=False,
                    session_id=None,
                    step_completed="full",
                    next_step=None,
                    next_command=None,
                    result=None,
                    message="Full mode failed at discovery step",
                    error=str(e),
                )
            else:
                # Need to determine which step failed based on session state
                session = self.session_store.load(session_id)
                if session:
                    if "red" not in session.steps_completed:
                        return SparringResponse(
                            success=False,
                            session_id=session_id,
                            step_completed="full",
                            next_step="red",
                            next_command=f'qwen_sparring(mode="red", session_id="{session_id}")',
                            result=None,
                            message="Full mode failed at red cell step",
                            error=str(e),
                        )
                    elif "blue" not in session.steps_completed:
                        return SparringResponse(
                            success=False,
                            session_id=session_id,
                            step_completed="full",
                            next_step="blue",
                            next_command=f'qwen_sparring(mode="blue", session_id="{session_id}")',
                            result=None,
                            message="Full mode failed at blue cell step",
                            error=str(e),
                        )
                    else:
                        return SparringResponse(
                            success=False,
                            session_id=session_id,
                            step_completed="full",
                            next_step="white",
                            next_command=f'qwen_sparring(mode="white", session_id="{session_id}")',
                            result=None,
                            message="Full mode failed at white cell step",
                            error=str(e),
                        )
                # Fallback
                return SparringResponse(
                    success=False,
                    session_id=session_id,
                    step_completed="full",
                    next_step=None,
                    next_command=None,
                    result=None,
                    message="Full mode failed",
                    error=str(e),
                )
    
    def _assemble_full_report(
        self,
        discovery: SparringResponse,
        red: SparringResponse,
        blue: SparringResponse,
        white: SparringResponse,
    ) -> str:
        """Assemble full session report from all 4 steps."""
        from qwen_mcp.engines.sparring_v2.formatters.output import ReportAssembler
        return ReportAssembler.assemble_full_report(
            self.session_store,
            discovery,
            red,
            blue,
            white,
        )
    
    # _report_progress inherited from ModeExecutor base class (with WebSocket broadcast)
