"""
Sparring Engine v2 - Flash Mode Executor

Fast 2-step analysis: Analyst → Drafter.
Uses ephemeral TTL checkpointing (300s) for fast mode support.

Refactored to inherit from BaseStageExecutor for stage-based execution with:
- Budget management (60s total budget for flash mode)
- Ephemeral TTL checkpointing (300s TTL)
- Automatic checkpointing after each stage
"""

import logging
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import Context

from qwen_mcp.engines.sparring_v2.base_stage_executor import (
    BaseStageExecutor, StageContext, StageResult
)
from qwen_mcp.engines.sparring_v2.models import SparringResponse
from qwen_mcp.engines.sparring_v2.config import (
    TIMEOUTS,
    MAX_TOKENS_CONFIG,
    MAX_THINKING_TOKENS_CONFIG,
    get_thinking_tokens_for_mode,
)
from qwen_mcp.prompts.sparring import FLASH_ANALYST_PROMPT, FLASH_DRAFTER_PROMPT

logger = logging.getLogger(__name__)


class FlashExecutor(BaseStageExecutor):
    """
    Execute flash mode: Analyst → Drafter with ephemeral TTL checkpointing.
    
    Inherits from BaseStageExecutor for:
    - Budget management (60s total budget)
    - Ephemeral TTL checkpointing (300s TTL)
    - Automatic checkpointing after each stage
    """
    
    STAGES = ["analyst", "drafter"]
    STAGE_WEIGHTS = {
        "analyst": 0.45,   # 40.5s (50% of 81s used)
        "drafter": 0.55,   # 49.5s (50% of 81s used)
    }
    EPHEMERAL_TTL = 300  # 300 seconds TTL for flash mode checkpoints
    
    def __init__(self, client, session_store):
        """Initialize FlashExecutor with 90s budget (increased from 60s to handle real API times)."""
        super().__init__(client, session_store, total_budget_seconds=90)
        self._topic = ""
        self._context = ""
        self._ctx = None
        self._ephemeral_session_id = None
    
    def get_stages(self) -> List[str]:
        """Get list of stages for flash mode."""
        return self.STAGES
    
    def get_stage_weights(self) -> Dict[str, float]:
        """Get budget weights for each stage."""
        return self.STAGE_WEIGHTS
    
    async def execute_stage(self, stage_name: str, context: StageContext) -> StageResult:
        """
        Execute a single stage.
        
        Args:
            stage_name: Name of the stage (analyst, drafter)
            context: Current stage context
            
        Returns:
            StageResult with execution outcome
        """
        logger.info(f"FlashExecutor executing stage: {stage_name}")
        
        try:
            if stage_name == "analyst":
                # Step 1: Analyst
                analyst_messages = [
                    {"role": "system", "content": FLASH_ANALYST_PROMPT},
                    {"role": "user", "content": f"Topic: {self._topic}\n\nContext:\n{self._context}"},
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
                    progress_callback=self._ctx.report_progress if self._ctx else None,
                )
                
                # Store analysis in context for drafter stage
                context.add_metadata("analyst_result", analysis)
                
                return StageResult(
                    stage_name=stage_name,
                    success=True,
                    result=analysis
                )
                
            elif stage_name == "drafter":
                # Step 2: Drafter - use analyst result from context
                analysis = context.get_metadata("analyst_result", "")
                
                drafter_messages = [
                    {"role": "system", "content": FLASH_DRAFTER_PROMPT},
                    {"role": "user", "content": f"Topic: {self._topic}\n\nContext: {self._context}\n\nAnalysis:\n{analysis}"},
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
                    progress_callback=self._ctx.report_progress if self._ctx else None,
                )
                
                return StageResult(
                    stage_name=stage_name,
                    success=True,
                    result=final_strategy
                )
            else:
                return StageResult(
                    stage_name=stage_name,
                    success=False,
                    error=f"Unknown stage: {stage_name}"
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
        Execute flash mode analysis with ephemeral TTL checkpointing.
        
        Args:
            topic: Topic for the sparring
            context: Additional context
            ctx: MCP context for progress reporting
            
        Returns:
            SparringResponse with analysis results
        """
        logger.info(f"Executing flash mode with ephemeral TTL checkpointing for topic: {topic}")
        
        try:
            # Store execution context for stage execution
            self._topic = topic
            self._context = context
            self._ctx = ctx
            
            # Create ephemeral session for checkpointing
            self._ephemeral_session_id = f"flash_{topic[:20].replace(' ', '_')}_{id(self)}"
            
            # Create initial stage context
            stage_context = StageContext(
                session_id=self._ephemeral_session_id,
                topic=topic,
                context=context
            )
            
            # Report progress: starting
            await self._report_progress(ctx, 0.0, "[Flash] Initializing Analysis...")
            
            # Execute all stages with recovery and TTL checkpointing
            results = await self.execute_with_recovery(stage_context)
            
            # Check for failures
            failed_stages = [name for name, r in results.items() if not r.success]
            
            if failed_stages:
                first_failed = failed_stages[0]
                return SparringResponse(
                    success=False,
                    session_id=None,
                    step_completed=first_failed,
                    next_step=first_failed,
                    next_command=None,
                    result=None,
                    message=f"Flash mode failed at stage: {first_failed}",
                    error=results[first_failed].error,
                )
            
            # Get final strategy from drafter stage
            final_strategy = results.get("drafter", StageResult("", False)).result
            output = self._format_output(final_strategy, "Flash Analysis")
            
            await self._report_progress(ctx, 100.0, "[Flash] Analysis Complete")
            
            return SparringResponse(
                success=True,
                session_id=None,  # Flash mode doesn't persist sessions
                step_completed="flash",
                next_step=None,
                next_command=None,
                result={"strategy": output},
                message="Flash analysis complete with ephemeral TTL checkpointing",
            )
            
        except Exception as e:
            logger.exception(f"Flash mode execution failed: {e}")
            return SparringResponse(
                success=False,
                session_id=None,
                step_completed="flash",
                next_step=None,
                next_command=None,
                result=None,
                message="Flash mode execution failed",
                error=str(e),
            )
    
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
