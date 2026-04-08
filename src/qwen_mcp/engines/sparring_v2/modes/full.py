"""
Sparring Engine v2 - Full Mode Executor

Execute complete sparring session in one call: discovery→red→blue→white.
Uses shorter word limits to ensure completion within MCP timeout.

Refactored to inherit from BaseStageExecutor for stage-based execution with:
- Budget management (225s total budget)
- Circuit breaker for failure recovery
- Automatic checkpointing after each stage
- White Cell regeneration loop re-enabled (max_loops=2)
"""

import logging
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import Context

from qwen_mcp.engines.sparring_v2.base_stage_executor import (
    BaseStageExecutor, StageContext, StageResult
)
from qwen_mcp.engines.sparring_v2.models import SparringResponse
from qwen_mcp.prompts.sparring import WORD_LIMITS

logger = logging.getLogger(__name__)


class FullExecutor(BaseStageExecutor):
    """
    Execute complete sparring session with stage-based execution.
    
    Inherits from BaseStageExecutor for:
    - Budget management (225s total budget)
    - Circuit breaker (3 failures → 60s recovery)
    - Automatic checkpointing after each stage
    - White Cell regeneration loop (max_loops=2)
    """
    
    STAGES = ["discovery", "red", "blue", "white"]
    STAGE_WEIGHTS = {
        "discovery": 0.15,  # 33.75s
        "red": 0.28,        # 63s
        "blue": 0.28,       # 63s
        "white": 0.29,      # 65.25s (includes regeneration budget)
    }
    
    def __init__(self, client, session_store):
        """Initialize FullExecutor with 225s budget."""
        super().__init__(client, session_store, total_budget_seconds=225)
        self._topic = ""
        self._context = ""
        self._ctx = None
        self._white_loop_count = 0
        self._max_white_loops = 2
    
    def get_stages(self) -> List[str]:
        """Get list of stages for full mode."""
        return self.STAGES
    
    def get_stage_weights(self) -> Dict[str, float]:
        """Get budget weights for each stage."""
        return self.STAGE_WEIGHTS
    
    async def execute_stage(self, stage_name: str, context: StageContext) -> StageResult:
        """
        Execute a single stage.
        
        Args:
            stage_name: Name of the stage (discovery, red, blue, white)
            context: Current stage context
            
        Returns:
            StageResult with execution outcome
        """
        logger.info(f"FullExecutor executing stage: {stage_name}")
        
        try:
            # Import the appropriate executor for this stage
            if stage_name == "discovery":
                from qwen_mcp.engines.sparring_v2.modes.discovery import DiscoveryExecutor
                executor = DiscoveryExecutor(self.client, self.session_store)
                result = await executor.execute(
                    topic=self._topic,
                    context=self._context,
                    ctx=self._ctx,
                    word_limit=WORD_LIMITS["full_discovery"],
                )
                # Store session_id from discovery
                if result.session_id:
                    context.session_id = result.session_id
                    
            elif stage_name == "red":
                from qwen_mcp.engines.sparring_v2.modes.red_cell import RedCellExecutor
                executor = RedCellExecutor(self.client, self.session_store)
                result = await executor.execute(
                    session_id=context.session_id,
                    ctx=self._ctx,
                    word_limit=WORD_LIMITS["full_red"],
                )
                
                # Debug: Log red stage result
                logger.debug(f"Red stage result: success={result.success}, error={result.error}, result keys={result.result.keys() if result.result else None}")
                if result.success and result.result:
                    logger.debug(f"Red critique present: {'critique' in result.result}")
                
            elif stage_name == "blue":
                from qwen_mcp.engines.sparring_v2.modes.blue_cell import BlueCellExecutor
                executor = BlueCellExecutor(self.client, self.session_store)
                result = await executor.execute(
                    session_id=context.session_id,
                    ctx=self._ctx,
                    word_limit=WORD_LIMITS["full_blue"],
                )
                
            elif stage_name == "white":
                from qwen_mcp.engines.sparring_v2.modes.white_cell import WhiteCellExecutor
                executor = WhiteCellExecutor(self.client, self.session_store)
                # Re-enable regeneration loop for White Cell (max 2 loops)
                result = await executor.execute(
                    session_id=context.session_id,
                    ctx=self._ctx,
                    allow_regeneration=True,
                    word_limit=WORD_LIMITS["full_white"],
                )
            else:
                return StageResult(
                    stage_name=stage_name,
                    success=False,
                    error=f"Unknown stage: {stage_name}"
                )
            
            # Check if stage succeeded
            if result.success:
                return StageResult(
                    stage_name=stage_name,
                    success=True,
                    result=result.result
                )
            else:
                return StageResult(
                    stage_name=stage_name,
                    success=False,
                    error=result.error or f"Stage {stage_name} failed"
                )
                
        except Exception as e:
            logger.exception(f"Stage {stage_name} raised exception: {e}")
            return StageResult(
                stage_name=stage_name,
                success=False,
                error=str(e)
            )
    
    async def execute(
        self,
        topic: str,
        context: str = "",
        ctx: Optional[Context] = None,
    ) -> SparringResponse:
        """
        Execute full sparring session with stage-based execution.
        
        Args:
            topic: Topic for the sparring
            context: Additional context
            ctx: MCP context for progress reporting
            
        Returns:
            SparringResponse with full session results
        """
        logger.info(f"Executing full mode with stage-based execution for topic: {topic}")
        
        try:
            # Store execution context for stage execution
            self._topic = topic
            self._context = context
            self._ctx = ctx
            
            # Create initial stage context
            stage_context = StageContext(
                session_id="",  # Will be set by discovery stage
                topic=topic,
                context=context
            )
            
            # Report progress: starting
            await self._report_progress(ctx, 0.0, "[Full] Starting stage-based execution...")
            
            # Execute all stages with recovery
            results = await self.execute_with_recovery(stage_context)
            
            # Get session_id from discovery stage
            session_id = stage_context.session_id
            
            # Check for failures
            failed_stages = [name for name, r in results.items() if not r.success]
            
            if failed_stages:
                # Find first failed stage for next_command
                first_failed = failed_stages[0]
                completed = self.STAGES[:self.STAGES.index(first_failed)]
                last_completed = completed[-1] if completed else None
                
                return SparringResponse(
                    success=False,
                    session_id=session_id if session_id else None,
                    step_completed=last_completed or "full",
                    next_step=first_failed,
                    next_command=f'qwen_sparring(mode="{first_failed}", session_id="{session_id}")' if session_id else f'qwen_sparring(mode="discovery", topic="{topic}")',
                    result=None,
                    message=f"Full mode failed at stage: {first_failed}",
                    error=results[first_failed].error,
                )
            
            # All stages completed successfully - assemble final report
            final_report = self._assemble_full_report(results)
            
            await self._report_progress(ctx, 100.0, "[Full] All stages completed successfully")
            
            return SparringResponse(
                success=True,
                session_id=session_id,
                step_completed="full",
                next_step=None,
                next_command=None,
                result={"full_report": final_report},
                message="Full sparring session complete with stage-based execution",
            )
            
        except Exception as e:
            logger.exception(f"Full mode execution failed: {e}")
            return SparringResponse(
                success=False,
                session_id=None,
                step_completed="full",
                next_step=None,
                next_command=None,
                result=None,
                message="Full mode execution failed",
                error=str(e),
            )
    
    def _assemble_full_report(self, results: Dict[str, StageResult]) -> str:
        """Assemble full session report from all stage results."""
        from qwen_mcp.engines.sparring_v2.formatters.output import ReportAssembler
        
        # Convert StageResult to SparringResponse-like format for assembler
        class MockSparringResponse:
            def __init__(self, result, success=True, session_id=None, error=None):
                self.result = result
                self.success = success
                self.session_id = session_id
                self.error = error
        
        discovery_mock = MockSparringResponse(
            results.get("discovery", StageResult("", False)).result,
            results.get("discovery", StageResult("", False)).success
        )
        red_mock = MockSparringResponse(
            results.get("red", StageResult("", False)).result,
            results.get("red", StageResult("", False)).success
        )
        blue_mock = MockSparringResponse(
            results.get("blue", StageResult("", False)).result,
            results.get("blue", StageResult("", False)).success
        )
        white_mock = MockSparringResponse(
            results.get("white", StageResult("", False)).result,
            results.get("white", StageResult("", False)).success
        )
        
        return ReportAssembler.assemble_full_report(
            self.session_store,
            discovery_mock,
            red_mock,
            blue_mock,
            white_mock,
        )
