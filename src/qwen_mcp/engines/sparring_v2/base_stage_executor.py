"""
Sparring Engine v2 - Base Stage Executor

This module provides the base class for stage-based execution with:
- BudgetManager: Dynamic timeout allocation with remaining budget tracking
- CircuitBreaker: Failure threshold (3 failures) with recovery timeout (60s)
- StageResult/StageContext dataclasses
- BaseStageExecutor: Abstract class with execute_with_recovery() method
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Type
from datetime import datetime, timezone
from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)


# =============================================================================
# Budget Manager - Dynamic Timeout Allocation
# =============================================================================

class BudgetManager:
    """
    Manages timeout budget allocation across stages.
    
    Features:
    - Dynamic timeout allocation based on stage weights
    - Remaining budget tracking
    - Per-stage budget enforcement
    """
    
    def __init__(self, total_budget_seconds: int, stage_weights: Optional[Dict[str, float]] = None):
        """
        Initialize budget manager.
        
        Args:
            total_budget_seconds: Total timeout budget in seconds
            stage_weights: Dictionary mapping stage names to weights (must sum to 1.0)
        """
        self.total_budget = total_budget_seconds
        self.stage_weights = stage_weights or {}
        self.start_time = time.time()
        self.stage_usage: Dict[str, float] = {}
        
    def get_stage_budget(self, stage_name: str) -> int:
        """
        Get allocated budget for a specific stage.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Budget in seconds for this stage
        """
        weight = self.stage_weights.get(stage_name, 1.0 / max(len(self.stage_weights), 1))
        return int(self.total_budget * weight)
    
    def record_usage(self, stage_name: str, seconds_used: float) -> None:
        """
        Record time used by a stage.
        
        Args:
            stage_name: Name of the stage
            seconds_used: Time consumed in seconds
        """
        self.stage_usage[stage_name] = seconds_used
        logger.debug(f"Stage '{stage_name}' used {seconds_used:.2f}s")
    
    def get_remaining_budget(self) -> float:
        """
        Get remaining budget in seconds.
        
        Returns:
            Remaining seconds in budget
        """
        elapsed = time.time() - self.start_time
        return max(0, self.total_budget - elapsed)
    
    def get_elapsed(self) -> float:
        """Get total elapsed time since budget start."""
        return time.time() - self.start_time
    
    def is_over_budget(self) -> bool:
        """Check if total budget is exhausted."""
        return self.get_remaining_budget() <= 0


# =============================================================================
# Dynamic Budget Manager - Extended Budget Management
# =============================================================================

class DynamicBudgetManager(BudgetManager):
    """
    Extended BudgetManager with dynamic budget adjustments.
    
    Features:
    - Time borrowing across stages
    - Timeout extension for complex tasks
    - Complexity-based budget adjustment
    """
    
    def __init__(
        self,
        total_budget_seconds: int,
        stage_weights: Optional[Dict[str, float]] = None,
        allow_borrow: bool = False,
        extend_timeout_pct: float = 0.5
    ):
        """
        Initialize DynamicBudgetManager.
        
        Args:
            total_budget_seconds: Total timeout budget in seconds
            stage_weights: Dictionary mapping stage names to weights
            allow_borrow: Whether to allow borrowing time from previous stages
            extend_timeout_pct: Percentage to extend timeout for complex tasks (0.5 = 50%)
        """
        super().__init__(total_budget_seconds, stage_weights)
        self.allow_borrow = allow_borrow
        self.extend_timeout_pct = extend_timeout_pct
        self.completed_stages: List[str] = []
        self.stage_order: List[str] = []
        self.complexity_factors: Dict[str, float] = {}
    
    def register_stage_order(self, stage_order: List[str]) -> None:
        """
        Register the order of stages for borrowing calculations.
        
        Args:
            stage_order: List of stage names in execution order
        """
        self.stage_order = stage_order
    
    def mark_stage_completed(self, stage_name: str) -> None:
        """
        Mark a stage as completed for borrowing purposes.
        
        Args:
            stage_name: Name of the completed stage
        """
        if stage_name not in self.completed_stages:
            self.completed_stages.append(stage_name)
    
    def set_stage_complexity(self, stage_name: str, complexity_factor: float = 1.0) -> None:
        """
        Set complexity factor for a stage.
        
        Args:
            stage_name: Name of the stage
            complexity_factor: Multiplier for complexity (1.0 = normal, >1.0 = complex)
        """
        self.complexity_factors[stage_name] = max(1.0, complexity_factor)
    
    def borrow_time_from_previous_stages(self, stage_name: str) -> float:
        """
        Calculate available borrowed time from completed previous stages.
        
        Args:
            stage_name: Current stage name to borrow time for
            
        Returns:
            Available time that can be borrowed from previous stages
        """
        if not self.allow_borrow or not self.stage_order:
            return 0.0
        
        if stage_name not in self.stage_order:
            return 0.0
        
        current_index = self.stage_order.index(stage_name)
        if current_index == 0:
            return 0.0
        
        available_borrowed_time = 0.0
        
        for i in range(current_index):
            previous_stage = self.stage_order[i]
            if previous_stage in self.completed_stages and previous_stage in self.stage_usage:
                # Calculate unused time from previous stage
                stage_budget = self.get_stage_budget(previous_stage)
                used = self.stage_usage.get(previous_stage, 0)
                unused = stage_budget - used
                if unused > 0:
                    available_borrowed_time += unused
        
        return max(0.0, available_borrowed_time)
    
    def extend_current_stage_timeout(self, base_timeout: float, is_complex: bool = False) -> float:
        """
        Apply timeout extension for current stage.
        
        Args:
            base_timeout: Base timeout in seconds
            is_complex: Whether the task is complex and needs extension
            
        Returns:
            Extended timeout value
        """
        if not is_complex:
            return base_timeout
        
        extension = base_timeout * self.extend_timeout_pct
        return base_timeout + extension
    
    def adjust_for_complexity(self, stage_name: str, base_budget: int, is_complex: bool = False) -> int:
        """
        Adjust token budget based on stage complexity.
        
        Args:
            stage_name: Name of the stage
            base_budget: Base token budget
            is_complex: Whether the stage is complex
            
        Returns:
            Adjusted token budget
        """
        if not is_complex:
            return base_budget
        
        complexity_factor = self.complexity_factors.get(stage_name, 1.5)
        return int(base_budget * complexity_factor)
    
    def get_dynamic_stage_budget(
        self,
        stage_name: str,
        base_budget: int,
        is_complex: bool = False
    ) -> int:
        """
        Get dynamic budget with all adjustments applied.
        
        Args:
            stage_name: Name of the stage
            base_budget: Base token budget
            is_complex: Whether the stage is complex
            
        Returns:
            Adjusted budget including borrowing and complexity adjustments
        """
        adjusted = self.adjust_for_complexity(stage_name, base_budget, is_complex)
        borrowed = int(self.borrow_time_from_previous_stages(stage_name))
        return adjusted + borrowed


# =============================================================================
# Circuit Breaker - Failure Recovery
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker pattern for stage execution.
    
    Features:
    - Failure threshold (default: 3 failures)
    - Recovery timeout (default: 60s)
    - Automatic state management (CLOSED → OPEN → HALF_OPEN → CLOSED)
    """
    
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = self.CLOSED
        
    def record_success(self) -> None:
        """Record successful execution - reset circuit."""
        self.failure_count = 0
        self.state = self.CLOSED
        logger.debug("Circuit breaker: SUCCESS - reset to CLOSED")
    
    def record_failure(self) -> None:
        """Record failure - potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = self.OPEN
            logger.warning(f"Circuit breaker: OPENED after {self.failure_count} failures")
        else:
            logger.debug(f"Circuit breaker: Failure {self.failure_count}/{self.failure_threshold}")
    
    def can_execute(self) -> bool:
        """
        Check if execution is allowed.
        
        Returns:
            True if circuit is CLOSED or HALF_OPEN (recovery attempted)
        """
        if self.state == self.CLOSED:
            return True
        
        if self.state == self.OPEN:
            # Check if recovery timeout has elapsed
            if self.last_failure_time is None:
                return True
            
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = self.HALF_OPEN
                logger.info("Circuit breaker: HALF_OPEN - attempting recovery")
                return True
            
            return False
        
        # HALF_OPEN - allow one attempt
        return True
    
    def reset(self) -> None:
        """Force reset circuit breaker to CLOSED state."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = self.CLOSED


# =============================================================================
# Stage Result and Context
# =============================================================================

@dataclass
class StageResult:
    """
    Result from executing a single stage.
    
    Attributes:
        stage_name: Name of the executed stage
        success: Whether the stage completed successfully
        result: Stage-specific result data
        error: Error message if failed
        execution_time_seconds: Time taken to execute
        retry_count: Number of retries attempted
    """
    stage_name: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_seconds: float = 0.0
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage_name": self.stage_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time_seconds": self.execution_time_seconds,
            "retry_count": self.retry_count
        }


@dataclass
class StageContext:
    """
    Context passed between stages.
    
    Attributes:
        session_id: Session identifier
        topic: Sparring topic
        context: Additional context
        stage_results: Results from previously executed stages
        metadata: Additional stage-specific metadata
        ctx: MCP Context object for progress reporting
        word_limit: Word limit for the stage
        allow_regeneration: Whether to allow regeneration (for white cell)
    """
    session_id: Optional[str]
    topic: str
    context: str = ""
    stage_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ctx: Optional[Any] = None
    word_limit: Optional[int] = None
    allow_regeneration: bool = True
    
    def get_stage_result(self, stage_name: str) -> Optional[Any]:
        """Get result from a previously executed stage."""
        return self.stage_results.get(stage_name)
    
    def add_stage_result(self, stage_name: str, result: Any) -> None:
        """Store result from executed stage."""
        self.stage_results[stage_name] = result
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to context."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata from context."""
        return self.metadata.get(key, default)


# =============================================================================
# Base Stage Executor - Abstract Base Class
# =============================================================================

class BaseStageExecutor(ABC):
    """
    Abstract base class for stage-based sparring execution.
    
    Provides:
    - Budget management with dynamic allocation
    - Circuit breaker for failure recovery
    - Stage checkpointing
    - Recovery from failed stages
    
    Subclasses must implement:
    - get_stages(): List of stage names to execute
    - execute_stage(): Execute a single stage
    - get_stage_weights(): Budget weights for each stage
    """
    
    def __init__(self, client, session_store, total_budget_seconds: int = 225):
        """
        Initialize base stage executor.
        
        Args:
            client: DashScopeClient for API calls
            session_store: SessionStore for checkpointing
            total_budget_seconds: Total timeout budget (default: 225s for full sparring)
        """
        self.client = client
        self.session_store = session_store
        self.total_budget_seconds = total_budget_seconds
        self.budget_manager: Optional[BudgetManager] = None
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
    
    async def _report_progress(self, ctx: Optional[Context], progress: float, message: str) -> None:
        """
        Safe progress reporting with dual broadcast (MCP + WebSocket for Roo Code HUD).
        
        Args:
            ctx: MCP context for progress reporting
            progress: Progress percentage (0-100)
            message: Status message
        """
        if ctx:
            try:
                # MCP progress notification (protocol compliance)
                await ctx.report_progress(progress=float(progress), message=message)
            except Exception:
                pass
        
        # WebSocket broadcast for Roo Code HUD visibility
        try:
            from qwen_mcp.specter.telemetry import get_broadcaster
            await get_broadcaster().broadcast_state({
                "operation": message,
                "progress": float(progress),
                "is_live": True
            }, project_id="sparring")
        except Exception:
            pass
        
    @abstractmethod
    def get_stages(self) -> List[str]:
        """
        Get list of stage names to execute.
        
        Returns:
            List of stage names in execution order
        """
        pass
    
    @abstractmethod
    async def execute_stage(self, stage_name: str, context: StageContext) -> StageResult:
        """
        Execute a single stage.
        
        Args:
            stage_name: Name of the stage to execute
            context: Current stage context
            
        Returns:
            StageResult with execution outcome
        """
        pass
    
    @abstractmethod
    def get_stage_weights(self) -> Dict[str, float]:
        """
        Get budget weights for each stage.
        
        Returns:
            Dictionary mapping stage names to weights (should sum to 1.0)
        """
        pass
    
    async def execute_with_recovery(
        self,
        context: StageContext,
        skip_stages: Optional[List[str]] = None
    ) -> Dict[str, StageResult]:
        """
        Execute all stages with recovery support.
        
        Features:
        - Budget-aware execution (skips stages if over budget)
        - Circuit breaker (skips stages if circuit is OPEN)
        - Checkpointing after each successful stage
        - Recovery from failed stages
        - Skip already completed stages (for session resumption)
        
        Args:
            context: Initial stage context
            skip_stages: Optional list of stage names to skip (already completed)
            
        Returns:
            Dictionary mapping stage names to StageResult
        """
        logger.info(f"Starting stage-based execution for session: {context.session_id!r}, skip_stages: {skip_stages}")
        
        # Initialize budget manager
        self.budget_manager = BudgetManager(
            total_budget_seconds=self.total_budget_seconds,
            stage_weights=self.get_stage_weights()
        )
        logger.info(f"BudgetManager initialized: total_budget={self.total_budget_seconds}s, remaining={self.budget_manager.get_remaining_budget():.2f}s")
        
        stages = self.get_stages()
        results: Dict[str, StageResult] = {}
        
        # Load existing session results for skipped stages
        if skip_stages and context.session_id:
            existing_session = self.session_store.load(context.session_id)
            if existing_session:
                # Handle both 'results' and 'step_results' attribute names
                results_attr = getattr(existing_session, 'results', None) or \
                              getattr(existing_session, 'step_results', None)
                if results_attr:
                    for stage_name in skip_stages:
                        if stage_name in results_attr:
                            results[stage_name] = StageResult(
                                stage_name=stage_name,
                                success=True,
                                result=results_attr[stage_name]
                            )
                            logger.info(f"Loaded existing result for stage: {stage_name}")
        
        for stage_name in stages:
            logger.info(f"Executing stage: {stage_name}, context.session_id before: {context.session_id!r}")
            # Skip already completed stages
            if skip_stages and stage_name in skip_stages:
                logger.info(f"Skipping already completed stage: {stage_name}")
                continue
            
            # Check budget
            remaining = self.budget_manager.get_remaining_budget()
            stage_budget = self.budget_manager.get_stage_budget(stage_name)
            logger.info(f"Budget check before {stage_name}: remaining={remaining:.2f}s, stage_budget={stage_budget}s, is_over={self.budget_manager.is_over_budget()}")
            
            if self.budget_manager.is_over_budget():
                logger.warning(f"Budget exhausted before stage '{stage_name}'")
                results[stage_name] = StageResult(
                    stage_name=stage_name,
                    success=False,
                    error="Budget exhausted",
                    execution_time_seconds=self.budget_manager.get_elapsed()
                )
                continue
            
            # Check circuit breaker
            if not self.circuit_breaker.can_execute():
                logger.warning(f"Circuit breaker OPEN - skipping stage '{stage_name}'")
                results[stage_name] = StageResult(
                    stage_name=stage_name,
                    success=False,
                    error="Circuit breaker OPEN",
                    execution_time_seconds=self.budget_manager.get_elapsed()
                )
                continue
            
            # Execute stage
            stage_start = time.time()
            logger.info(f"Executing stage: {stage_name}")
            
            try:
                stage_result = await self.execute_stage(stage_name, context)
                stage_time = time.time() - stage_start
                stage_result.execution_time_seconds = stage_time
                
                # Log session_id after stage execution
                logger.info(f"Stage {stage_name} completed, context.session_id after: {context.session_id!r}")
                
                # Record budget usage
                self.budget_manager.record_usage(stage_name, stage_time)
                
                if stage_result.success:
                    self.circuit_breaker.record_success()
                    context.add_stage_result(stage_name, stage_result.result)
                    
                    # Checkpoint after successful stage
                    await self._checkpoint_stage(context, stage_name, stage_result)
                    logger.info(f"Stage '{stage_name}' completed successfully in {stage_time:.2f}s")
                else:
                    self.circuit_breaker.record_failure()
                    logger.warning(f"Stage '{stage_name}' failed: {stage_result.error}")
                
                results[stage_name] = stage_result
                
            except Exception as e:
                stage_time = time.time() - stage_start
                self.circuit_breaker.record_failure()
                self.budget_manager.record_usage(stage_name, stage_time)
                
                error_result = StageResult(
                    stage_name=stage_name,
                    success=False,
                    error=str(e),
                    execution_time_seconds=stage_time
                )
                results[stage_name] = error_result
                logger.exception(f"Stage '{stage_name}' raised exception: {e}")
        
        return results
    
    async def _checkpoint_stage(self, context: StageContext, stage_name: str, result: StageResult) -> None:
        """
        Save checkpoint after successful stage execution.
        
        Args:
            context: Current stage context
            stage_name: Name of completed stage
            result: Stage execution result
        """
        try:
            # Load existing checkpoint or create new
            checkpoint = self.session_store.load(context.session_id)
            
            # Extract the actual result data from StageResult.result
            # This ensures get_step_result() can find the data (e.g., "critique" key)
            result_data = result.result if result.result else result.to_dict()
            
            if checkpoint:
                # Update existing checkpoint
                if stage_name not in checkpoint.steps_completed:
                    checkpoint.steps_completed.append(stage_name)
                checkpoint.current_step = stage_name
                checkpoint.results[stage_name] = result_data
                checkpoint.updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                self.session_store.save(checkpoint)
            else:
                # Create new checkpoint
                checkpoint = self.session_store.create_session(
                    topic=context.topic,
                    context=context.context
                )
                checkpoint.session_id = context.session_id
                checkpoint.steps_completed = [stage_name]
                checkpoint.current_step = stage_name
                checkpoint.results[stage_name] = result_data
                self.session_store.save(checkpoint)
            
            logger.debug(f"Checkpoint saved for stage '{stage_name}'")
            
        except Exception as e:
            logger.warning(f"Failed to save checkpoint for stage '{stage_name}': {e}")
            # Non-fatal - continue execution
