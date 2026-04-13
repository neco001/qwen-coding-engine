"""
Sparring Engine v2 - Backward Compatibility Wrappers

These wrapper classes maintain the existing interface for Flash/Full/Pro executors
while delegating to UnifiedSparringExecutor internally. This ensures no breaking
changes for existing code that depends on these specific executor classes.

The wrappers:
- FlashExecutor: Maintains existing FlashExecutor interface
- FullExecutor: Maintains existing FullExecutor interface  
- ProExecutor: Maintains existing ProExecutor interface

All wrappers use UnifiedSparringExecutor internally for the actual execution.
"""

import logging
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import Context

from qwen_mcp.engines.sparring_v2.base_stage_executor import (
    BaseStageExecutor, StageContext, StageResult
)
from qwen_mcp.engines.sparring_v2.models import SparringResponse
from qwen_mcp.engines.sparring_v2.config import get_mode_profile
from qwen_mcp.engines.sparring_v2.modes.unified import UnifiedSparringExecutor

logger = logging.getLogger(__name__)


class FlashExecutor(BaseStageExecutor):
    """
    Backward compatibility wrapper for Flash mode.
    
    Maintains the existing FlashExecutor interface while delegating to
    UnifiedSparringExecutor internally.
    
    This class exists to ensure no breaking changes for code that imports
    FlashExecutor directly. All functionality is delegated to UnifiedSparringExecutor.
    """
    
    STAGES = ["analyst", "drafter"]
    STAGE_WEIGHTS = {
        "analyst": 0.45,
        "drafter": 0.55,
    }
    EPHEMERAL_TTL = 300  # 300 seconds TTL for flash mode checkpoints
    
    def __init__(self, client, session_store):
        """Initialize FlashExecutor with 90s budget."""
        # Store original parameters for reference
        self._client = client
        self._session_store = session_store
        self._topic = ""
        self._context = ""
        self._ctx = None
        
        # Create unified executor internally
        self._unified_executor = UnifiedSparringExecutor(
            client=client,
            session_store=session_store,
            mode="flash",
            total_budget_seconds=90,  # Keep original budget
        )
    
    def get_stages(self) -> List[str]:
        """Get list of stages for flash mode."""
        return self.STAGES
    
    def get_stage_weights(self) -> Dict[str, float]:
        """Get budget weights for each stage."""
        return self.STAGE_WEIGHTS
    
    async def execute_stage(self, stage_name: str, context: StageContext) -> StageResult:
        """Execute a single stage using the unified executor."""
        return await self._unified_executor.execute_stage(stage_name, context)
    
    async def execute(
        self,
        topic: str,
        context: str = "",
        ctx: Optional[Context] = None,
        word_limit: Optional[int] = None,
        thinking_tokens: Optional[int] = None,
        **kwargs,
    ) -> SparringResponse:
        """Execute the full sparring session using unified executor."""
        self._topic = topic
        self._context = context
        self._ctx = ctx
        
        # Execute using unified executor
        response = await self._unified_executor.execute(
            topic=topic,
            context=context,
            ctx=ctx,
            word_limit=word_limit,
            thinking_tokens=thinking_tokens,
            **kwargs,
        )
        
        return response
    
    def _format_output(self, raw: str, label: str) -> str:
        """Format output for flash mode."""
        return self._unified_executor._format_output(raw, label)


class FullExecutor(BaseStageExecutor):
    """
    Backward compatibility wrapper for Full mode.
    
    Maintains the existing FullExecutor interface while delegating to
    UnifiedSparringExecutor internally.
    
    This class exists to ensure no breaking changes for code that imports
    FullExecutor directly. All functionality is delegated to UnifiedSparringExecutor.
    """
    
    STAGES = ["discovery", "red", "blue", "white"]
    STAGE_WEIGHTS = {
        "discovery": 0.15,
        "red": 0.28,
        "blue": 0.28,
        "white": 0.29,
    }
    
    def __init__(self, client, session_store):
        """Initialize FullExecutor with 225s budget."""
        # Store original parameters for reference
        self._client = client
        self._session_store = session_store
        self._topic = ""
        self._context = ""
        self._ctx = None
        
        # Create unified executor internally
        self._unified_executor = UnifiedSparringExecutor(
            client=client,
            session_store=session_store,
            mode="full",
            total_budget_seconds=225,  # Keep original budget
        )
    
    def get_stages(self) -> List[str]:
        """Get list of stages for full mode."""
        return self.STAGES
    
    def get_stage_weights(self) -> Dict[str, float]:
        """Get budget weights for each stage."""
        return self.STAGE_WEIGHTS
    
    async def execute_stage(self, stage_name: str, context: StageContext) -> StageResult:
        """Execute a single stage using the unified executor."""
        return await self._unified_executor.execute_stage(stage_name, context)
    
    async def execute(
        self,
        topic: str,
        context: str = "",
        ctx: Optional[Context] = None,
        word_limit: Optional[int] = None,
        thinking_tokens: Optional[int] = None,
        **kwargs,
    ) -> SparringResponse:
        """Execute the full sparring session using unified executor."""
        self._topic = topic
        self._context = context
        self._ctx = ctx
        
        # Execute using unified executor
        response = await self._unified_executor.execute(
            topic=topic,
            context=context,
            ctx=ctx,
            word_limit=word_limit,
            thinking_tokens=thinking_tokens,
            **kwargs,
        )
        
        return response
    
    def _assemble_full_report(self, results: Dict[str, StageResult]) -> str:
        """Assemble full session report from all stage results."""
        return self._unified_executor._assemble_full_report(results)


class ProExecutor(BaseStageExecutor):
    """
    Backward compatibility wrapper for Pro mode.
    
    Maintains the existing ProExecutor interface while delegating to
    UnifiedSparringExecutor internally.
    
    This class exists to ensure no breaking changes for code that imports
    ProExecutor directly. All functionality is delegated to UnifiedSparringExecutor.
    """
    
    STAGES = ["discovery", "red", "blue", "white"]
    STAGE_WEIGHTS = {
        "discovery": 0.15,
        "red": 0.28,
        "blue": 0.28,
        "white": 0.29,
    }
    
    def __init__(self, client, session_store):
        """Initialize ProExecutor with 900s budget (225s per stage)."""
        # Store original parameters for reference
        self._client = client
        self._session_store = session_store
        self._topic = ""
        self._context = ""
        self._ctx = None
        
        # Create unified executor internally
        self._unified_executor = UnifiedSparringExecutor(
            client=client,
            session_store=session_store,
            mode="pro",
            total_budget_seconds=900,  # Keep original budget (225s per stage)
        )
    
    def get_stages(self) -> List[str]:
        """Get list of stages for pro mode."""
        return self.STAGES
    
    def get_stage_weights(self) -> Dict[str, float]:
        """Get budget weights for each stage."""
        return self.STAGE_WEIGHTS
    
    async def execute_stage(self, stage_name: str, context: StageContext) -> StageResult:
        """Execute a single stage using the unified executor."""
        return await self._unified_executor.execute_stage(stage_name, context)
    
    async def execute(
        self,
        topic: str,
        context: str = "",
        ctx: Optional[Context] = None,
        word_limit: Optional[int] = None,
        thinking_tokens: Optional[int] = None,
        **kwargs,
    ) -> SparringResponse:
        """Execute the full sparring session using unified executor."""
        self._topic = topic
        self._context = context
        self._ctx = ctx
        
        # Execute using unified executor
        response = await self._unified_executor.execute(
            topic=topic,
            context=context,
            ctx=ctx,
            word_limit=word_limit,
            thinking_tokens=thinking_tokens,
            **kwargs,
        )
        
        return response
    
    def _compile_full_report(self, results: Dict[str, StageResult]) -> str:
        """Compile full report from all stage results."""
        return self._unified_executor._compile_full_report(results)
