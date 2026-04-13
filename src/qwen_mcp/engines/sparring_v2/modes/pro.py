"""
Sparring Engine v2 - Pro Mode Executor (sparring3)

Execute sparring session step-by-step with checkpoints:
discovery→red→blue→white (EACH as SEPARATE MCP call).

KEY DESIGN: Each call executes ONLY the next stage, preventing timeout.
User calls qwen_sparring(mode="sparring3") repeatedly to progress through stages.

Uses higher word limits (800 words) and token budgets (4096 tokens) for deep analysis.
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
    Execute sparring3 step-by-step: ONE STAGE PER MCP CALL.

    Flow:
    1. Call 1: discovery → returns roles + "next: red"
    2. Call 2: red → returns critique + "next: blue"
    3. Call 3: blue → returns defense + "next: white"
    4. Call 4: white → returns consensus (FINAL)

    This prevents timeout because each stage gets its own 300s MCP timeout.
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
        Execute pro mode (sparring3) - ONE STAGE PER CALL.

        This is the KEY design: each MCP call executes only the next stage,
        preventing timeout. User calls sparring3 repeatedly to progress through stages.

        Flow:
        1. Call 1: discovery → returns roles + "next: red"
        2. Call 2: red → returns critique + "next: blue"
        3. Call 3: blue → returns defense + "next: white"
        4. Call 4: white → returns consensus (FINAL)

        Args:
            topic: Topic for the sparring
            context: Additional context
            ctx: MCP context for progress reporting
            session_id: Optional existing session ID to resume from

        Returns:
            SparringResponse with single stage result and next_command
        """
        logger.info(f"Executing pro mode (sparring3) - SINGLE STAGE for topic: {topic[:50]}..., session_id: {session_id}")

        try:
            # Initialize defaults
            self._topic = topic
            self._context = context
            self._ctx = ctx
            self._word_limit = WORD_LIMITS.get("pro", 800)

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
                        self._topic = existing_session.topic
                    if existing_session.context:
                        self._context = existing_session.context

            # Determine which stage to execute next
            next_stage = self._get_next_stage(completed_stages)
            
            if next_stage is None:
                # All stages completed - return full report from existing session
                if existing_session and existing_session.results:
                    full_report = self._compile_full_report_from_checkpoint(existing_session)
                    return SparringResponse(
                        success=True,
                        session_id=session_id,
                        step_completed="pro",
                        next_step=None,
                        next_command=None,
                        result=full_report,
                        message="Pro mode (sparring3) - all stages already completed",
                    )
                else:
                    return SparringResponse(
                        success=False,
                        session_id=session_id,
                        step_completed="pro",
                        next_step=None,
                        next_command=None,
                        result=None,
                        message="No next stage to execute and no completed results available",
                        error="Session has no next stage and no completed results",
                    )

            # Execute ONLY the next stage (not all stages!)
            stage_context = StageContext(
                session_id=session_id,
                topic=self._topic,
                context=self._context
            )

            await self._report_progress(ctx, 0.0, f"[Pro] Executing {next_stage} stage...")

            # Execute the single stage
            stage_result = await self.execute_stage(next_stage, stage_context)
            
            # Update session_id if discovery just created it
            if next_stage == "discovery" and stage_result.success and stage_context.session_id:
                session_id = stage_context.session_id

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
                    message=f"Pro mode failed at stage: {next_stage}",
                    error=stage_result.error,
                )

            # Stage succeeded - return with next_command
            next_stage_after = self._get_stage_after(next_stage)
            
            await self._report_progress(ctx, 100.0, f"[Pro] {next_stage.title()} complete")

            if next_stage_after:
                # More stages to go - return next_command
                return SparringResponse(
                    success=True,
                    session_id=session_id,
                    step_completed=next_stage,
                    next_step=next_stage_after,
                    next_command=f'qwen_sparring(mode="{next_stage_after}", session_id="{session_id}")',
                    result=stage_result.result,
                    message=f"Pro mode: {next_stage} complete. Next: {next_stage_after}",
                )
            else:
                # All stages complete - compile full report
                # Load the full session to get all results
                full_session = self.session_store.load(session_id) if session_id else None
                if full_session and full_session.results:
                    full_report = self._compile_full_report_from_checkpoint(full_session)
                else:
                    full_report = stage_result.result

                return SparringResponse(
                    success=True,
                    session_id=session_id,
                    step_completed=next_stage,
                    next_step=None,
                    next_command=None,
                    result=full_report,
                    message="Pro mode (sparring3) completed successfully - all stages done",
                )

        except Exception as e:
            logger.error(f"Pro mode execution failed: {e}", exc_info=True)
            return SparringResponse(
                success=False,
                session_id=session_id,
                step_completed="pro",
                next_step=None,
                next_command=None,
                result=None,
                message="Pro mode execution failed",
                error=str(e),
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

    def _compile_full_report_from_checkpoint(self, session) -> str:
        """Compile full report from a SessionCheckpoint."""
        report_parts = [
            "# 📊 Pełny Raport Sparring3 - Szczegółowa Analiza Krok-po-Kroku",
            "",
            f"**Session ID:** `{session.session_id}`",
            "**Tryb:** `sparring3 (pro)` - Analiza krok-po-kroku z checkpointami",
            "**Cel:** Głęboka analiza strategiczna z wysokim budżetem tokenów (4096 tokens/cell)",
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
            "**sparring3 (pro)** = Najbardziej szczegółowy tryb analizy z osobnymi wywołaniami MCP dla każdej komórki",
        ])

        return "\n".join(report_parts)
