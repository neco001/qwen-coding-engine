# unified.py
"""
Unified Sparring Executor - Single executor supporting all sparring modes.

This module provides a unified executor that supports all three sparring modes
(flash, full, pro) through a single implementation using MODE_PROFILES configuration.

Features:
- Single executor for all modes (flash/full/pro)
- Mode-specific configuration from MODE_PROFILES
- Dynamic budget management with time borrowing
- Backward compatibility with existing mode executors
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from qwen_mcp.engines.sparring_v2.models import SparringResponse
from dataclasses import dataclass

from qwen_mcp.engines.sparring_v2.base_stage_executor import (
    BaseStageExecutor,
    StageResult,
    StageContext,
    DynamicBudgetManager,
)
from qwen_mcp.engines.sparring_v2.config import MODE_PROFILES, get_mode_profile

logger = logging.getLogger(__name__)


class UnifiedSparringExecutor(BaseStageExecutor):
    """
    Unified executor supporting all sparring modes through configuration.
    
    This class consolidates the three sparring modes (flash, full, pro) into
    a single executor that uses MODE_PROFILES for mode-specific configuration.
    
    Features:
    - Single implementation for all modes
    - Dynamic budget management with time borrowing
    - Mode-specific word limits and thinking tokens
    - Backward compatibility with existing mode executors
    """
    
    def __init__(
        self,
        client,
        session_store,
        mode: str = "full",
        total_budget_seconds: Optional[int] = None,
        stage_weights: Optional[Dict[str, float]] = None,
        allow_borrow: bool = False,
        extend_timeout_pct: float = 0.5,
    ):
        """
        Initialize UnifiedSparringExecutor.
        
        Args:
            client: DashScopeClient for API calls
            session_store: SessionStore for checkpointing
            mode: Sparring mode ('flash', 'full', or 'pro')
            total_budget_seconds: Override total budget (optional)
            stage_weights: Override stage weights (optional)
            allow_borrow: Allow time borrowing from previous stages
            extend_timeout_pct: Percentage to extend timeout for complex tasks
        """
        # Get mode profile if not overridden
        if mode not in MODE_PROFILES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {list(MODE_PROFILES.keys())}")
        
        profile = get_mode_profile(mode)
        
        # Use profile values or overrides
        self._mode = mode
        self._profile = profile
        self._stages = profile.stages
        self._stage_weights = stage_weights or profile.stage_weights
        self._total_budget = total_budget_seconds or profile.total_budget
        
        # Initialize parent with budget manager
        super().__init__(client, session_store, total_budget_seconds=self._total_budget)
        
        # Initialize dynamic budget manager
        self.budget_manager = DynamicBudgetManager(
            total_budget_seconds=self._total_budget,
            stage_weights=self._stage_weights,
            allow_borrow=allow_borrow,
            extend_timeout_pct=extend_timeout_pct,
        )
        
        # Register stage order for borrowing calculations
        self.budget_manager.register_stage_order(self._stages)
        
        # Store mode-specific settings
        self._word_limits = profile.word_limits
        self._thinking_tokens = profile.thinking_tokens
        self._timeout_config = profile.timeout_config
        
        # Stage-specific executors (lazy-loaded)
        self._stage_executors: Dict[str, Any] = {}
    
    @property
    def mode(self) -> str:
        """Get the current sparring mode."""
        return self._mode
    
    @property
    def profile(self):
        """Get the mode profile."""
        return self._profile
    
    def get_stages(self) -> List[str]:
        """Get list of stages for current mode."""
        return self._stages
    
    def get_stage_weights(self) -> Dict[str, float]:
        """Get budget weights for each stage."""
        return self._stage_weights
    
    def get_word_limit(self, stage_name: str) -> int:
        """Get word limit for a specific stage."""
        return self._word_limits.get(stage_name, 150)
    
    def get_thinking_tokens(self, stage_name: str) -> int:
        """Get thinking tokens for a specific stage."""
        return self._thinking_tokens.get(stage_name, 1024)
    
    def get_stage_timeout(self, stage_name: str) -> float:
        """Get timeout for a specific stage."""
        return self._timeout_config.get(stage_name, 30.0)
    
    def get_dynamic_stage_budget(self, stage_name: str, is_complex: bool = False) -> int:
        """
        Get dynamic budget for a stage with all adjustments.
        
        Args:
            stage_name: Name of the stage
            is_complex: Whether the stage is complex
            
        Returns:
            Adjusted budget including borrowing and complexity adjustments
        """
        if not isinstance(self.budget_manager, DynamicBudgetManager):
            # Fallback to base class budget
            return self.budget_manager.get_stage_budget(stage_name)
        
        base_budget = self.budget_manager.get_stage_budget(stage_name)
        return self.budget_manager.get_dynamic_stage_budget(stage_name, base_budget, is_complex)
    
    async def execute_stage(self, stage_name: str, context: StageContext) -> StageResult:
        """
        Execute a single stage using the appropriate executor.
        
        Args:
            stage_name: Name of the stage to execute
            context: Current stage context
            
        Returns:
            StageResult with execution outcome
        """
        logger.info(f"UnifiedSparringExecutor ({self._mode}) executing stage: {stage_name}")
        
        try:
            # Get or create stage-specific executor
            stage_executor = self._get_stage_executor(stage_name)
            
            # Execute stage with correct parameters based on executor type
            # Each legacy executor has different signature, so we adapt the call
            # Use stage-specific word limit from mode profile
            stage_word_limit = self.get_word_limit(stage_name)
            
            if stage_name == "discovery":
                # DiscoveryExecutor.execute(topic, context, ctx, word_limit, session_id)
                # Pass context.session_id if it exists (for session resumption)
                result = await stage_executor.execute(
                    topic=context.topic,
                    context=context.context or "",
                    ctx=context.ctx,
                    word_limit=stage_word_limit,
                    session_id=context.session_id  # Pass existing session_id if available
                )
                
                # Update context.session_id from discovery result
                if result.success and hasattr(result, 'session_id') and result.session_id:
                    context.session_id = result.session_id
                    logger.info(f"Discovery created session: {context.session_id!r}")
            elif stage_name in ("red", "blue"):
                # RedCellExecutor/BlueCellExecutor.execute(session_id, ctx, word_limit)
                result = await stage_executor.execute(
                    session_id=context.session_id,
                    ctx=context.ctx,
                    word_limit=stage_word_limit
                )
            elif stage_name == "white":
                # WhiteCellExecutor.execute(session_id, ctx, allow_regeneration, word_limit)
                result = await stage_executor.execute(
                    session_id=context.session_id,
                    ctx=context.ctx,
                    allow_regeneration=context.allow_regeneration or True,
                    word_limit=stage_word_limit
                )
            elif stage_name in ("analyst", "drafter"):
                # FlashAnalystExecutor/DrafterExecutor may have different signatures
                result = await stage_executor.execute(
                    session_id=context.session_id,
                    ctx=context.ctx,
                    word_limit=stage_word_limit
                )
            else:
                # Fallback for unknown stages
                result = await stage_executor.execute(context=context)
            
            # Record budget usage
            if self.budget_manager:
                self.budget_manager.record_usage(stage_name, result.execution_time_seconds if hasattr(result, 'execution_time_seconds') else 0)
            
            # Mark stage as completed for borrowing
            if isinstance(self.budget_manager, DynamicBudgetManager):
                self.budget_manager.mark_stage_completed(stage_name)
            
            return result
            
        except Exception as e:
            logger.exception(f"Stage '{stage_name}' failed: {e}")
            return StageResult(
                stage_name=stage_name,
                success=False,
                error=str(e),
                execution_time_seconds=0,
            )
    
    def _get_stage_executor(self, stage_name: str):
        """
        Get or create executor for a specific stage.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Stage-specific executor instance
        """
        if stage_name in self._stage_executors:
            return self._stage_executors[stage_name]
        
        # Import stage-specific executors
        if stage_name == "discovery":
            from qwen_mcp.engines.sparring_v2.modes.discovery import DiscoveryExecutor
            executor_class = DiscoveryExecutor
        elif stage_name == "red":
            from qwen_mcp.engines.sparring_v2.modes.red_cell import RedCellExecutor
            executor_class = RedCellExecutor
        elif stage_name == "blue":
            from qwen_mcp.engines.sparring_v2.modes.blue_cell import BlueCellExecutor
            executor_class = BlueCellExecutor
        elif stage_name == "white":
            from qwen_mcp.engines.sparring_v2.modes.white_cell import WhiteCellExecutor
            executor_class = WhiteCellExecutor
        elif stage_name == "analyst":
            from qwen_mcp.engines.sparring_v2.modes.flash import FlashAnalystExecutor
            executor_class = FlashAnalystExecutor
        elif stage_name == "drafter":
            from qwen_mcp.engines.sparring_v2.modes.flash import FlashDrafterExecutor
            executor_class = FlashDrafterExecutor
        else:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        # Create executor instance
        executor = executor_class(self.client, self.session_store)
        self._stage_executors[stage_name] = executor
        
        return executor
    
    async def execute(
        self,
        topic: str,
        context: str = "",
        ctx=None,
        word_limit: Optional[int] = None,
        thinking_tokens: Optional[int] = None,
        **kwargs,
    ) -> SparringResponse:
        """
        Execute the full sparring session.

        Args:
            topic: Sparring topic
            context: Additional context
            ctx: MCP context for progress reporting
            word_limit: Override word limit (optional)
            thinking_tokens: Override thinking tokens (optional)
            **kwargs: Additional arguments

        Returns:
            SparringResponse with all stage results
        """
        try:
            logger.info(f"UnifiedSparringExecutor ({self._mode}) starting session for topic: {topic[:50]}...")

            # Store execution parameters
            self._topic = topic
            self._context = context
            self._ctx = ctx

            # Create initial context - use first stage's word limit if not overridden
            # session_id will be created by discovery stage
            initial_word_limit = word_limit if word_limit is not None else self.get_word_limit(self._stages[0])
            
            initial_context = StageContext(
                session_id=None,  # Discovery will create session_id
                topic=topic,
                context=context,
                ctx=ctx,
                word_limit=initial_word_limit,
                allow_regeneration=True,  # Enable regeneration for white cell
            )

            # Execute all stages
            results = await self.execute_with_recovery(initial_context)

            # Get session_id from context (updated by discovery stage)
            session_id = initial_context.session_id
            
            # Assemble final report
            report = self._assemble_full_report(results)

            # Save session checkpoint (fix missing session file issue)
            try:
                logger.debug(f"Attempting to save session checkpoint for {session_id}")
                response = SparringResponse.success(
                    session_id=session_id,
                    step="final",
                    next_step=None,
                    next_command=None,
                    result=results,
                    message="Unified sparring session completed successfully",
                    messages_appended=len(results)
                )
                logger.debug(f"Created SparringResponse: {response}")
                checkpoint = self._convert_response_to_checkpoint(response)
                logger.debug(f"Converted to SessionCheckpoint: {checkpoint}")
                logger.debug(f"Session checkpoint data: {checkpoint.to_dict()}")
                self.session_store.save(checkpoint)
                logger.info(f"Successfully saved session checkpoint for {session_id}")
            except AttributeError as e:
                logger.error(f"Failed to save session checkpoint (missing attribute): {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error saving session checkpoint: {e}", exc_info=True)
                raise  # Re-raise to ensure visibility

            # Return structured SparringResponse (fixes 'messages_appended' AttributeError)
            return SparringResponse.success(
                session_id=session_id,
                step="final",
                next_step=None,
                next_command=None,
                result=results,
                message="Unified sparring session completed successfully",
                messages_appended=len(results)
            )
        except Exception as e:
            logger.error(f"Unified sparring execution failed: {str(e)}", exc_info=True)
            session_id = self._topic[:8] + "-error" if hasattr(self, '_topic') else "error-" + str(uuid.uuid4())[:8]
            return SparringResponse.error(
                message=f"Unified sparring session failed for topic: {self._topic[:50]}...",
                error=str(e),
                session_id=session_id
            )
    
    def _convert_response_to_checkpoint(self, response: SparringResponse) -> SessionCheckpoint:
        """Convert SparringResponse to SessionCheckpoint for session_store.save() (anti-degeneration fix)"""
        from qwen_mcp.engines.session_store import SessionCheckpoint
        try:
            return SessionCheckpoint(
                session_id=response.session_id,
                topic=self._topic,
                context=self._context,
                status="completed",
                steps_completed=list(response.result.keys()) if response.result else [],
                current_step="final",
                results=response.result or {},
                messages=getattr(response, 'messages', []),
                has_stages=True,
                stage_count=len(response.result) if response.result else 4
            )
        except AttributeError as e:
            raise AttributeError(f"Missing required SparringResponse attribute for checkpoint: {e}")

    def _assemble_full_report(self, results: Dict[str, StageResult]) -> str:
        """
        Assemble full session report from all stage results.
        
        Args:
            results: Dictionary mapping stage names to StageResult
            
        Returns:
            Combined report string
        """
        report_parts = []
        
        for stage_name, result in results.items():
            if result.success and result.result:
                # Extract text from result
                if isinstance(result.result, dict):
                    text = result.result.get("text", str(result.result))
                else:
                    text = str(result.result)
                
                report_parts.append(f"## {stage_name.upper()}\n\n{text}")
        
        return "\n\n".join(report_parts)
