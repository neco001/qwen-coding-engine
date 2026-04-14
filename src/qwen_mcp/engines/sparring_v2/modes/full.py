"""
Sparring Engine v2 - Full Mode Executor (sparring2)

Execute complete sparring session in one call: discovery→red→blue→white.
ALL stages run sequentially in ONE MCP call (225s total budget).

⚠️ WARNING: This mode risks timeout if stages take too long!
For step-by-step execution, use sparring3 (pro mode) instead.

Uses shorter word limits to help fit within MCP timeout.

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
                logger.info(f"[Full.execute_stage.discovery] result.session_id={result.session_id!r}, result.success={result.success}")
                if result.session_id:
                    context.session_id = result.session_id
                    logger.info(f"[Full.execute_stage.discovery] Updated context.session_id to {context.session_id!r}")
                else:
                    logger.warning(f"[Full.execute_stage.discovery] result.session_id is falsy: {result.session_id!r}")
                    
            elif stage_name == "red":
                from qwen_mcp.engines.sparring_v2.modes.red_cell import RedCellExecutor
                executor = RedCellExecutor(self.client, self.session_store)
                logger.info(f"[Full.execute_stage.red] BEFORE: context.session_id={context.session_id!r}")
                result = await executor.execute(
                    session_id=context.session_id,
                    ctx=self._ctx,
                    word_limit=WORD_LIMITS["full_red"],
                )
                logger.info(f"[Full.execute_stage.red] AFTER: context.session_id={context.session_id!r}, result.session_id={result.session_id!r}")
                
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
        session_id: Optional[str] = None,
    ) -> SparringResponse:
        """
        Execute full sparring session with stage-based execution.
        
        Can operate in two modes:
        1. Fresh execution: No session_id provided, run all stages
        2. Step-by-step: session_id provided, run only next stage
        
        Args:
            topic: Topic for the sparring
            context: Additional context
            ctx: MCP context for progress reporting
            session_id: Optional existing session ID to resume from
            
        Returns:
            SparringResponse with session results
        """
        logger.info(f"Executing full mode (sparring2) - topic: {topic[:50]}..., session_id: {session_id}")
        
        # Normalize empty string session_id to None
        if session_id == "":
            session_id = None
        
        # Load existing session to determine next stage
        existing_session = None
        completed_stages = []
        
        if session_id:
            existing_session = self.session_store.load(session_id)
            if existing_session:
                logger.info(f"Resuming existing session: {session_id}")
                completed_stages = list(existing_session.steps_completed)
                logger.info(f"Already completed stages: {completed_stages}")
                
                # Restore topic/context from saved session
                if existing_session.topic:
                    topic = existing_session.topic
                if existing_session.context:
                    context = existing_session.context

        # Execute ALL stages sequentially in ONE MCP call (sparring2 = full batch mode)
        # Store execution context for stage execution
        self._topic = topic
        self._context = context
        self._ctx = ctx

        # Initialize stage context - session_id will be updated by discovery stage
        stage_context = StageContext(
            session_id=session_id,
            topic=topic,
            context=context
        )

        await self._report_progress(ctx, 0.0, "[Full] Starting full sparring session (all stages)...")

        # Execute ALL stages in sequence
        stages_to_run = self.get_stages()
        last_stage_result = None
        
        for next_stage in stages_to_run:
            # Skip already completed stages (for session resumption)
            if next_stage in completed_stages:
                logger.info(f"Skipping already completed stage: {next_stage}")
                continue
            
            await self._report_progress(ctx, 0.0, f"[Full] Executing {next_stage} stage...")

            # Execute the stage
            stage_result = await self.execute_stage(next_stage, stage_context)
            
            # Update session_id if discovery just created it
            if next_stage == "discovery" and stage_result.success and stage_context.session_id:
                session_id = stage_context.session_id
                completed_stages.append(next_stage)
            
            # Handle stage failure
            if not stage_result.success:
                next_stage_after_fail = self._get_stage_after(next_stage)
                return SparringResponse(
                    success=False,
                    session_id=session_id,
                    step_completed=next_stage,
                    next_step=next_stage_after_fail,
                    next_command=f'qwen_sparring(mode="{next_stage_after_fail or "discovery"}", session_id="{session_id}")' if session_id else None,
                    result=None,
                    message=f"Full mode failed at stage: {next_stage}",
                    error=stage_result.error,
                )
            
            # Mark stage as completed
            completed_stages.append(next_stage)
            last_stage_result = stage_result
            
            await self._report_progress(ctx, 50.0 + (len(completed_stages) / len(stages_to_run)) * 50.0, f"[Full] {next_stage.title()} complete")

        # All stages complete - compile full report
        # Load the full session to get all results
        full_session = self.session_store.load(session_id) if session_id else None
        if full_session and full_session.results:
            full_report = self._compile_full_report_from_checkpoint(full_session)
        else:
            full_report = self._compile_full_report_from_stages(stages_to_run, last_stage_result)

        return SparringResponse(
            success=True,
            session_id=session_id,
            step_completed="full",
            next_step=None,
            next_command=None,
            result=full_report,
            message="Full mode (sparring2) completed successfully - all stages done",
        )

    def _get_next_stage(self, completed_stages: List[str]) -> Optional[str]:
        """Get the next stage to execute based on completed stages."""
        for stage in self.STAGES:
            if stage not in completed_stages:
                return stage
        return None  # All stages completed

    def _get_stage_after(self, current_stage: str) -> Optional[str]:
        """Get the stage that follows the current stage."""
        try:
            idx = self.STAGES.index(current_stage)
            if idx + 1 < len(self.STAGES):
                return self.STAGES[idx + 1]
        except ValueError:
            pass
        return None  # Current is last stage

    def _compile_full_report_from_stages(self, stages: List[str], last_result: Any) -> str:
        """Compile full report from stage results when session checkpoint is not available."""
        report_parts = [
            "# 📊 Pełny Raport Sparring2 - Analiza Wszystkich Etapów",
            "",
            "**Tryb:** `sparring2 (full)` - Wszystkie etapy w jednym wywołaniu",
            "",
            "---",
            "",
        ]
        
        # Add results from each stage
        for stage_name in stages:
            stage_titles = {
                "discovery": "🎯 Krok 1: Discovery - Definicja Ról",
                "red": "🔴 Krok 2: Red Cell - Audyt Ryzyk",
                "blue": "🔵 Krok 3: Blue Cell - Obrona Strategiczna",
                "white": "⚪ Krok 4: White Cell - Synteza i Rekomendacje",
            }
            
            report_parts.append(f"## {stage_titles.get(stage_name, stage_name)}")
            report_parts.append("")
            report_parts.append(f"*Stage {stage_name} completed*")
            report_parts.append("")
            report_parts.append("---")
            report_parts.append("")
        
        return "\n".join(report_parts)

    def _compile_full_report_from_checkpoint(self, session) -> str:
        """Compile full report from a SessionCheckpoint."""
        report_parts = [
            "# 📊 Pełny Raport Sparring2 - Analiza Wszystkich Etapów",
            "",
            f"**Session ID:** `{session.session_id}`",
            "**Tryb:** `sparring2 (full)` - Analiza wszystkich etapów w jednym wywołaniu",
            "**Cel:** Kompleksowa analiza z użyciem mniejszego budżetu słów na etap",
            "",
            f"**Etap:** {', '.join(session.steps_completed)}",
            "",
            "---",
            "",
        ]

        # Add roles if available
        if session.roles:
            report_parts.append("### 🎭 Wybrane Role:")
            report_parts.append(f"- **Red:** {session.roles.get('red_role', 'N/A')}")
            report_parts.append(f"- **Blue:** {session.roles.get('blue_role', 'N/A')}")
            report_parts.append(f"- **White:** {session.roles.get('white_role', 'N/A')}")
            report_parts.append("")
            report_parts.append("---")
            report_parts.append("")

        # Add results from each stage
        for stage_name in ["discovery", "red", "blue", "white"]:
            if stage_name in session.results:
                result = session.results[stage_name]
                if isinstance(result, dict):
                    # Extract the main content
                    content = result.get("critique") or result.get("defense") or \
                              result.get("consensus") or result.get("roles", {}) or \
                              str(result)
                else:
                    content = str(result)

                stage_titles = {
                    "discovery": "🎯 Krok 1: Discovery - Definicja Ról",
                    "red": "🔴 Krok 2: Red Cell - Audyt Ryzyk",
                    "blue": "🔵 Krok 3: Blue Cell - Obrona Strategiczna",
                    "white": "⚪ Krok 4: White Cell - Synteza i Rekomendacje",
                }

                report_parts.append(f"## {stage_titles.get(stage_name, stage_name)}")
                report_parts.append("")
                report_parts.append(content)
                report_parts.append("")
                report_parts.append("---")
                report_parts.append("")

        # Summary table
        report_parts.extend([
            "## 📋 Podsumowanie Procesu",
            "",
            f"**Etap:** {', '.join(session.steps_completed)}",
            f"**Status:** {session.status}",
            "",
            "**sparring2 (full)** = Tryb kompleksowej analizy - wszystkie etapy w jednym wywołaniu lub krok-po-kroku",
        ])

        return "\n".join(report_parts)
            
    
