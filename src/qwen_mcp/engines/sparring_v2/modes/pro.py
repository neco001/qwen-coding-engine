"""
Sparring Engine v2 - Pro Mode Executor (sparring3)

Execute sparring session step-by-step with checkpoints:
discovery→red→blue→white (each as separate MCP call).
Uses higher word limits (800 words) and token budgets (4096 tokens) for deep analysis.

Refactored to inherit from BaseStageExecutor for stage-based execution with:
- Budget management
- Circuit breaker for failure recovery
- Automatic checkpointing
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


class ProExecutor(BaseStageExecutor):
    """
    Execute sparring session step-by-step with checkpoints (sparring3).
    
    Inherits from BaseStageExecutor for:
    - Budget management (225s total budget)
    - Circuit breaker (3 failures → 60s recovery)
    - Automatic checkpointing after each stage
    """
    
    STAGES = ["discovery", "red", "blue", "white"]
    STAGE_WEIGHTS = {
        "discovery": 0.15,  # 33.75s
        "red": 0.28,        # 63s
        "blue": 0.28,       # 63s
        "white": 0.29,      # 65.25s
    }
    
    def __init__(self, client, session_store):
        """Initialize ProExecutor with 225s budget."""
        super().__init__(client, session_store, total_budget_seconds=225)
        self._topic = ""
        self._context = ""
        self._ctx = None
        self._word_limit = 800
    
    def get_stages(self) -> List[str]:
        """Get list of stages for pro mode."""
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
        logger.info(f"ProExecutor executing stage: {stage_name}")
        
        try:
            # Import the appropriate executor for this stage
            if stage_name == "discovery":
                from qwen_mcp.engines.sparring_v2.modes.discovery import DiscoveryExecutor
                executor = DiscoveryExecutor(self.client, self.session_store)
                result = await executor.execute(
                    topic=self._topic,
                    context=self._context,
                    ctx=self._ctx,
                    word_limit=self._word_limit,
                )
                # CRITICAL: Update context.session_id with the newly created session
                logger.info(f"Discovery result: success={result.success}, session_id={result.session_id!r}")
                if result.success and result.session_id:
                    context.session_id = result.session_id
                    logger.info(f"✓ Discovery updated context.session_id to: {context.session_id!r}")
                else:
                    logger.warning(f"✗ Discovery did NOT update session_id (success={result.success}, session_id={result.session_id!r})")
            elif stage_name == "red":
                # Validate session_id before passing to RedCellExecutor
                logger.info(f"Red stage checking context.session_id: {context.session_id!r}")
                if not context.session_id:
                    logger.error(f"✗ Red stage FAILED: No session_id available after discovery stage")
                    return StageResult(
                        stage_name=stage_name,
                        success=False,
                        error="No session_id available after discovery stage"
                    )
                from qwen_mcp.engines.sparring_v2.modes.red_cell import RedCellExecutor
                executor = RedCellExecutor(self.client, self.session_store)
                logger.info(f"✓ Red stage executing with session_id: {context.session_id!r}")
                result = await executor.execute(
                    session_id=context.session_id,
                    ctx=self._ctx,
                    word_limit=self._word_limit,
                )
            elif stage_name == "blue":
                from qwen_mcp.engines.sparring_v2.modes.blue_cell import BlueCellExecutor
                executor = BlueCellExecutor(self.client, self.session_store)
                result = await executor.execute(
                    session_id=context.session_id,
                    ctx=self._ctx,
                    word_limit=self._word_limit,
                )
            elif stage_name == "white":
                from qwen_mcp.engines.sparring_v2.modes.white_cell import WhiteCellExecutor
                executor = WhiteCellExecutor(self.client, self.session_store)
                result = await executor.execute(
                    session_id=context.session_id,
                    ctx=self._ctx,
                    word_limit=self._word_limit,
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
        Execute pro mode sparring - step-by-step with checkpoints.
        
        Args:
            topic: Topic for the sparring
            context: Additional context
            ctx: MCP context for progress reporting
            session_id: Optional existing session ID to resume from
            
        Returns:
            SparringResponse with step results and next_command for continuation
        """
        logger.info(f"Executing pro mode (sparring3) for topic: {topic}, session_id: {session_id}")
        
        try:
            # Initialize defaults BEFORE checking existing session
            self._topic = topic
            self._context = context
            self._ctx = ctx
            self._word_limit = WORD_LIMITS.get("pro", 800)
            
            # Normalize empty string session_id to None for consistent handling
            if session_id == "":
                session_id = None
            
            # Check if we should resume an existing session
            existing_session = None
            if session_id:
                existing_session = self.session_store.load(session_id)
                if existing_session:
                    logger.info(f"Resuming existing session: {session_id}")
                    # Override with saved values if available
                    if existing_session.topic:
                        self._topic = existing_session.topic
                    if existing_session.context:
                        self._context = existing_session.context
            
            # Create initial stage context
            # CRITICAL FIX: Use None instead of "" for new sessions (empty string is falsy)
            stage_context = StageContext(
                session_id=session_id,  # None for new session
                topic=self._topic,
                context=self._context
            )
            logger.info(f"Initial stage_context.session_id: {stage_context.session_id!r}")
            
            # Report progress: starting
            await self._report_progress(ctx, 0.0, "[Pro] Starting stage-based execution...")
            
            # Check if resuming an existing session with completed stages
            completed_stages = []
            if existing_session and hasattr(existing_session, 'steps_completed'):
                # steps_completed is a List[str], not a dict
                completed_stages = list(existing_session.steps_completed)
                logger.info(f"Session {session_id} has completed stages: {completed_stages}")
            
            # Execute all stages with recovery (will skip completed stages)
            results = await self.execute_with_recovery(stage_context, skip_stages=completed_stages)
            
            # Get session_id from stage context (may be from existing session or new discovery)
            session_id = stage_context.session_id or session_id
            
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
                    step_completed=last_completed or "pro",
                    next_step=first_failed,
                    next_command=f'qwen_sparring(mode="{first_failed}", session_id="{session_id}")' if session_id else f'qwen_sparring(mode="discovery", topic="{topic}")',
                    result=None,
                    message=f"Pro mode failed at stage: {first_failed}",
                    error=results[first_failed].error,
                )
            
            # All stages completed successfully - compile full report
            full_report = self._compile_full_report(results)
            
            await self._report_progress(ctx, 100.0, "[Pro] All stages completed successfully")
            
            return SparringResponse(
                success=True,
                session_id=session_id,
                step_completed="pro",
                next_step=None,
                next_command=None,
                result=full_report,
                message="Pro mode (sparring3) completed successfully with full step-by-step analysis",
                error=None,
            )
            
        except Exception as e:
            logger.error(f"Pro mode execution failed: {e}", exc_info=True)
            return SparringResponse(
                success=False,
                session_id=None,
                step_completed="pro",
                next_step=None,
                next_command=None,
                result=None,
                message="Pro mode execution failed",
                error=str(e),
            )
    
    def _compile_full_report(self, results: Dict[str, StageResult]) -> str:
        """Compile full report from all stage results."""
        report_parts = [
            "# 📊 Pełny Raport Sparring3 - Szczegółowa Analiza Krok-po-Kroku",
            "",
            "**Tryb:** `sparring3 (pro)` - Analiza krok-po-kroku z checkpointami",
            "**Cel:** Głęboka analiza strategiczna z wysokim budżetem tokenów (4096 tokens/cell)",
            "",
            "---",
            "",
        ]
        
        # Helper to extract text from result
        def extract_text(result) -> str:
            if not result:
                return ""
            if isinstance(result, str):
                return result
            if isinstance(result, dict):
                for key in ["critique", "defense", "consensus", "strategy", "roles"]:
                    if key in result:
                        val = result[key]
                        return str(val) if not isinstance(val, dict) else str(val)
                return str(result)
            return str(result)
        
        # Helper to count words safely
        def count_words(text) -> int:
            if not text:
                return 0
            if isinstance(text, str):
                return len(text.split())
            return 0
        
        # Stage display names
        stage_names = {
            "discovery": ("🎯 Krok 1: Discovery - Definicja Ról", results.get("discovery")),
            "red": ("🔴 Krok 2: Red Cell - Audyt Ryzyk", results.get("red")),
            "blue": ("🔵 Krok 3: Blue Cell - Obrona Strategiczna", results.get("blue")),
            "white": ("⚪ Krok 4: White Cell - Synteza i Rekomendacje", results.get("white")),
        }
        
        for stage_name, (display_name, result) in stage_names.items():
            text = extract_text(result.result if result else None)
            if text:
                report_parts.append(f"## {display_name}")
                report_parts.append("")
                report_parts.append(text)
                report_parts.append("")
                report_parts.append("---")
                report_parts.append("")
        
        # Add summary table
        report_parts.extend([
            "## 📋 Podsumowanie Procesu",
            "",
            "| Krok | Rola | Status | Długość odpowiedzi |",
            "|------|------|--------|-------------------|",
            f"| 1 | Discovery | {'✅' if results.get('discovery', StageResult('', False)).success else '❌'} | ~{count_words(extract_text(results.get('discovery', StageResult('', False)).result if results.get('discovery') else None))} słów |",
            f"| 2 | Red Cell | {'✅' if results.get('red', StageResult('', False)).success else '❌'} | ~{count_words(extract_text(results.get('red', StageResult('', False)).result if results.get('red') else None))} słów |",
            f"| 3 | Blue Cell | {'✅' if results.get('blue', StageResult('', False)).success else '❌'} | ~{count_words(extract_text(results.get('blue', StageResult('', False)).result if results.get('blue') else None))} słów |",
            f"| 4 | White Cell | {'✅' if results.get('white', StageResult('', False)).success else '❌'} | ~{count_words(extract_text(results.get('white', StageResult('', False)).result if results.get('white') else None))} słów |",
            "",
            "**sparring3 (pro)** = Najbardziej szczegółowy tryb analizy z osobnymi wywołaniami MCP dla każdej komórki",
        ])
        
        return "\n".join(report_parts)
